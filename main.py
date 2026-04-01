import io
import os

import streamlit as st

# Load .env when running locally (no-op on Streamlit Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils import (
    extract_video_id,
    fetch_transcript_with_whisper_fallback,
    summarize_with_gemini,          # alias for summarize_with_llama
    fetch_video_metadata,
    generate_key_points,
    generate_quiz,
    export_summary_as_txt,
    segments_to_srt,
    translate_segments_to_srt,
    translate_transcript_to_language,
    WHISPER_LANGUAGES,
)


_GTTS_LANG_MAP: dict[str, str] = {
    "English": "en", "Bengali": "bn", "Hindi": "hi",
    "Spanish": "es", "French": "fr", "German": "de",
    "Japanese": "ja", "Arabic": "ar", "Portuguese": "pt",
    "Russian": "ru", "Chinese": "zh",
}


def _strip_srt(text: str) -> str:
    """Remove all timestamp formats and SRT metadata — keep only spoken text.

    Handles three formats that can appear in translated_captions:
      1. SRT sequence numbers     — bare integers on their own line
      2. SRT arrow timestamps     — 00:00:01,000 --> 00:00:04,000
      3. Bracket timestamps       — [00:04] or [01:23:45] (YouTube captions path)
    """
    import re
    # 1. SRT sequence numbers (bare integer lines)
    cleaned = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # 2. SRT arrow timestamp lines
    cleaned = re.sub(
        r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    # 3. Inline bracket timestamps like [00:04] or [01:23:45] anywhere in a line
    cleaned = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", cleaned)
    # Collapse leftover blank lines / extra whitespace into single spaces
    cleaned = re.sub(r"\n{2,}", " ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _text_to_mp3(text: str, lang_label: str) -> tuple[bytes | None, str | None]:
    """Convert text to MP3 bytes using gTTS.
    Automatically strips SRT formatting before synthesis.
    Returns (mp3_bytes, None) on success, (None, error_message) on failure."""
    try:
        from gtts import gTTS
    except ImportError:
        return None, "gTTS is not installed. Run: `pip install gtts`"
    try:
        clean_text = _strip_srt(text)
        if not clean_text:
            return None, "No speakable text found after stripping captions."
        lang_code = _GTTS_LANG_MAP.get(lang_label, "en")
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), None
    except Exception as e:
        return None, str(e)


def _get_secret(key: str) -> str | None:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Summarizer",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg-primary:   #0a0a0f;
    --bg-card:      #111118;
    --bg-elevated:  #16161f;
    --accent:       #7c3aed;
    --accent-glow:  #7c3aed55;
    --accent-light: #a78bfa;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e1e2e;
    --border-hover: #7c3aed66;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
  }

  [data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  #MainMenu, footer, header { visibility: hidden; }

  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .stTextInput > div > div > input,
  .stTextArea > div > textarea,
  .stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s, box-shadow 0.2s !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 0 18px var(--accent-glow) !important;
  }

  .stDownloadButton > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent-light) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 1.1rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--accent-light) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  .streamlit-expanderHeader {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
  }

  [data-testid="metric-container"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-family: 'Space Mono', monospace !important;
  }

  .stAlert { border-radius: 8px !important; border-left-width: 3px !important; }
  .stSpinner > div { border-top-color: var(--accent) !important; }

  .yt-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }
  .yt-card:hover { border-color: var(--border-hover); }

  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-light), #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
  }

  .badge {
    display: inline-block;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--accent-light);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
  }

  .source-badge-yt      { color: #10b981 !important; }
  .source-badge-whisper { color: #f59e0b !important; }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
  }

  pre, code {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--accent-light) !important;
    font-family: 'Space Mono', monospace !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ▶ YT Summarizer")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    api_key    = _get_secret("GROQ_API_KEY")
    proxy_user = _get_secret("WEBSHARE_USER")
    proxy_pass = _get_secret("WEBSHARE_PASS")

    if api_key:
        st.success("API key loaded ✓", icon="🔑")
    else:
        st.error("GROQ_API_KEY missing. Add to .env or Streamlit secrets.", icon="⚠")

    if proxy_user and proxy_pass:
        st.success("Proxy loaded ✓", icon="🌐")
    else:
        st.info("No proxy — fine for local use.", icon="💻")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**⚙️ Summary Settings**")

    summary_style = st.selectbox(
        "Summary style",
        ["Concise", "Detailed", "Bullet Points", "Executive Brief"],
        index=0,
    )
    summary_language = st.selectbox(
        "Output language",
        ["English", "Bengali", "Hindi", "Spanish", "French", "German", "Japanese"],
        index=0,
    )
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=128)
    include_timestamps = st.toggle("Include timestamps in transcript", value=False)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**🎙️ Whisper Fallback Settings**")
    st.caption(
        "Used automatically when a video has no captions. "
        "Requires **yt-dlp** (`pip install yt-dlp`)."
    )
    whisper_lang_label = st.selectbox(
        "Audio language hint for Whisper",
        list(WHISPER_LANGUAGES.keys()),
        index=0,
    )
    whisper_language_code = WHISPER_LANGUAGES[whisper_lang_label]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**🌐 Caption Translation**")
    caption_translate_lang = st.selectbox(
        "Translate captions to",
        ["None", "Bengali", "English", "Hindi", "Spanish", "French", "German", "Japanese"],
        index=0,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**🍪 YouTube Cookies**")
    st.caption(
        "If transcript fetching is blocked, upload your `cookies.txt` "
        "(Netscape format). Export with the "
        "[Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) "
        "Chrome extension while logged into YouTube."
    )

    cookies_txt: str | None = None
    uploaded_cookies = st.file_uploader(
        "cookies.txt (optional)",
        type=["txt"],
        label_visibility="collapsed",
    )
    if uploaded_cookies is not None:
        try:
            cookies_txt = uploaded_cookies.read().decode("utf-8", errors="ignore")
            if cookies_txt.strip():
                st.success("Cookies loaded ✓", icon="🍪")
            else:
                st.warning("Uploaded file appears empty.", icon="⚠")
                cookies_txt = None
        except Exception as exc:
            st.error(f"Could not read cookies file: {exc}", icon="⚠")
            cookies_txt = None
    else:
        secret_cookies = _get_secret("YT_COOKIES")
        if secret_cookies and secret_cookies.strip():
            cookies_txt = secret_cookies
            st.success("Cookies loaded from env/secrets ✓", icon="🍪")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<span class="badge">Llama 3.3-70B</span>'
        '<span class="badge">Whisper v3</span>'
        '<span class="badge">v2.0.0</span>',
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">YouTube Summarizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    'Paste any YouTube URL → AI-powered insights (Llama 3.3-70B + Whisper v3). '
    'Works even with no captions!'
    '</p>',
    unsafe_allow_html=True,
)

col_input, col_btn, col_refresh = st.columns([5, 1, 1], vertical_alignment="bottom")
with col_input:
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("⚡ Analyse", use_container_width=True)
with col_refresh:
    if st.button("🔄 Reset", use_container_width=True):
        for key in ["summary", "key_points", "quiz", "transcript", "video_id",
                    "meta", "whisper_segments", "transcript_source",
                    "translated_captions", "translated_srt", "caption_lang"]:
            st.session_state.pop(key, None)
        st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Processing ────────────────────────────────────────────────────────────────
if run:
    if not api_key:
        st.error("No API key found. Add GROQ_API_KEY to your .env file.")
        st.stop()
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(url)
    if not video_id:
        st.error("Could not parse a valid YouTube video ID from that URL.")
        st.stop()

    # Metadata
    with st.spinner("Fetching video metadata…"):
        meta = fetch_video_metadata(video_id)

    if meta:
        m1, m2, m3, m4 = st.columns(4)
        raw_title = meta.get("title", "—")
        m1.metric("▶ Title",    raw_title[:30] + "…" if len(raw_title) > 30 else raw_title)
        m2.metric("👤 Channel", meta.get("author",   "—"))
        m3.metric("⏱ Duration", meta.get("duration", "—"))
        m4.metric("👁 Views",   meta.get("views",    "—"))

        thumb = meta.get("thumbnail")
        if thumb:
            with st.expander("🖼 Thumbnail preview", expanded=False):
                st.image(thumb, use_container_width=False, width=480)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Transcript — with Whisper fallback
    with st.spinner(
        "Extracting transcript… (if no captions are found, "
        "Whisper will automatically download & transcribe the audio)"
    ):
        transcript, transcript_error, whisper_segments, transcript_source = (
            fetch_transcript_with_whisper_fallback(
                video_id,
                api_key=api_key,
                include_timestamps=include_timestamps,
                cookies_txt=cookies_txt,
                proxy_user=proxy_user,
                proxy_pass=proxy_pass,
                whisper_language=whisper_language_code,
            )
        )

    if transcript_error:
        st.error(f"Transcript error: {transcript_error}")
        st.stop()

    # Show source badge
    if transcript_source == "whisper":
        st.info(
            "🎙️ **Whisper transcription** — no captions were available, "
            "so the audio was downloaded and transcribed automatically.",
            icon="🤖",
        )
    else:
        st.success("✅ YouTube captions found and loaded.", icon="▶")

    st.session_state["transcript"]        = transcript
    st.session_state["video_id"]          = video_id
    st.session_state["meta"]              = meta or {}
    st.session_state["whisper_segments"]  = whisper_segments
    st.session_state["transcript_source"] = transcript_source

    # Summary
    with st.spinner("Generating summary… (long videos may take a moment)"):
        summary = summarize_with_gemini(
            transcript=transcript,
            api_key=api_key,
            style=summary_style,
            language=summary_language,
            max_tokens=max_tokens,
        )

    # Key points — reuse summary to avoid re-summarizing the full transcript
    with st.spinner("Extracting key points…"):
        key_points = generate_key_points(transcript, api_key, max_tokens, summary=summary, language=summary_language)

    # Quiz — reuse summary to avoid re-summarizing the full transcript
    with st.spinner("Building quiz questions…"):
        quiz = generate_quiz(transcript, api_key, summary=summary, language=summary_language)

    # Caption translation (if requested)
    translated_captions: str | None = None
    translated_srt:      str | None = None

    if caption_translate_lang != "None":
        with st.spinner(f"Translating captions to {caption_translate_lang}…"):
            if whisper_segments:
                # Whisper path — we have timestamps → produce translated SRT
                translated_srt = translate_segments_to_srt(
                    whisper_segments, caption_translate_lang, api_key
                )
                translated_captions = translated_srt          # show same text in tab
            else:
                # YouTube captions path — plain text translation
                translated_captions = translate_transcript_to_language(
                    transcript, caption_translate_lang, api_key
                )
                # Also produce a minimal SRT from translated text (no timestamps)
                # We'll just offer the plain translated text for download

    st.session_state["summary"]             = summary
    st.session_state["key_points"]          = key_points
    st.session_state["quiz"]                = quiz
    st.session_state["translated_captions"] = translated_captions
    st.session_state["translated_srt"]      = translated_srt
    st.session_state["caption_lang"]        = caption_translate_lang


# ── Results ───────────────────────────────────────────────────────────────────
if "summary" in st.session_state:
    summary             = st.session_state["summary"]
    key_points          = st.session_state["key_points"]
    quiz                = st.session_state["quiz"]
    transcript          = st.session_state["transcript"]
    whisper_segments    = st.session_state.get("whisper_segments")
    transcript_source   = st.session_state.get("transcript_source", "youtube")
    translated_captions = st.session_state.get("translated_captions")
    translated_srt      = st.session_state.get("translated_srt")
    caption_lang        = st.session_state.get("caption_lang", "None")

    # Build tab list dynamically
    tab_labels = ["📝 Summary", "🎯 Key Points", "❓ Quiz", "📄 Transcript"]
    show_captions_tab = translated_captions is not None
    if show_captions_tab:
        tab_labels.append(f"🌐 Captions ({caption_lang})")

    tabs = st.tabs(tab_labels)

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="yt-card">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown('</div>', unsafe_allow_html=True)

        txt_export = export_summary_as_txt(
            title=st.session_state.get("meta", {}).get("title", "Video"),
            summary=summary,
            key_points=key_points,
        )
        st.download_button(
            "⬇ Download Summary (.txt)",
            data=txt_export,
            file_name="summary.txt",
            mime="text/plain",
        )

    # ── Tab 2: Key Points ─────────────────────────────────────────────────────
    with tabs[1]:
        if isinstance(key_points, list):
            for i, point in enumerate(key_points, 1):
                st.markdown(
                    f'<div class="yt-card"><span class="badge">{i}</span> {point}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="yt-card">', unsafe_allow_html=True)
            st.markdown(key_points)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 3: Quiz ───────────────────────────────────────────────────────────
    with tabs[2]:
        import json as _json, re as _re

        # Normalise: if quiz came back as a raw JSON string, parse it
        _quiz = quiz
        if isinstance(_quiz, str):
            try:
                _clean = _re.sub(r"```(?:json)?\s*|\s*```", "", _quiz).strip()
                _parsed = _json.loads(_clean)
                if isinstance(_parsed, list):
                    _quiz = _parsed
            except Exception:
                pass  # leave as string, will render below

        if isinstance(_quiz, list) and _quiz:
            for i, item in enumerate(_quiz, 1):
                with st.expander(f"Q{i}: {item.get('question', 'Question')}", expanded=False):
                    st.markdown(f"**Answer:** {item.get('answer', '—')}")
                    if item.get("explanation"):
                        st.caption(item["explanation"])
        else:
            st.info("Quiz could not be parsed. Raw output:", icon="⚠️")
            st.code(quiz, language="json")

    # ── Tab 4: Transcript ─────────────────────────────────────────────────────
    with tabs[3]:
        # Source info
        if transcript_source == "whisper":
            st.info("🎙️ This transcript was generated by Groq Whisper (no YouTube captions were available).")
        else:
            st.success("▶ Source: YouTube captions")

        with st.expander("Full transcript", expanded=False):
            st.text_area("", value=transcript, height=420, label_visibility="collapsed")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇ Download Transcript (.txt)",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain",
            )

        # SRT download — available when Whisper produced segments
        if whisper_segments:
            with dl_col2:
                srt_content = segments_to_srt(whisper_segments)
                st.download_button(
                    "⬇ Download Original Captions (.srt)",
                    data=srt_content,
                    file_name="captions_original.srt",
                    mime="text/plain",
                )

    # ── Tab 5: Translated Captions (dynamic) ─────────────────────────────────
    if show_captions_tab:
        with tabs[4]:
            if translated_srt:
                st.info(
                    f"🌐 Translated to **{caption_lang}** · with timestamps (SRT)",
                    icon="✅",
                )
            else:
                st.warning(
                    f"🌐 Translated to **{caption_lang}** · plain text only "
                    "(timestamps are only available when Whisper transcription was used — "
                    "this video had YouTube captions which don't carry timing info)",
                    icon="ℹ️",
                )

            # Preview
            preview = translated_captions[:3000]
            if len(translated_captions) > 3000:
                preview += "\n\n… (truncated — download the full file below)"
            st.text_area(
                "Translated captions preview",
                value=preview,
                height=320,
                label_visibility="collapsed",
            )

            # ── Download buttons ──────────────────────────────────────────────
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    "⬇ Download Captions (.txt)",
                    data=translated_captions,
                    file_name=f"captions_{caption_lang.lower()}.txt",
                    mime="text/plain",
                )
            if translated_srt:
                with dl_c2:
                    st.download_button(
                        "⬇ Download Captions (.srt)",
                        data=translated_srt,
                        file_name=f"captions_{caption_lang.lower()}.srt",
                        mime="text/plain",
                    )

            # ── AI Audio (gTTS) ───────────────────────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("**🔊 Generate AI Audio**")
            st.caption(
                "Convert the translated captions to spoken MP3 audio using Google TTS. "
                "Long captions may take a moment to generate."
            )

            if caption_lang not in _GTTS_LANG_MAP:
                st.warning(f"Audio generation is not supported for {caption_lang} yet.", icon="⚠️")
            else:
                audio_key = f"mp3_audio_{caption_lang}"

                # Reserve a fixed slot for the audio player ABOVE the button
                # so it fills in-place instead of pushing content downward
                audio_slot = st.empty()

                # Pre-fill the slot if audio already exists in session state
                if audio_key in st.session_state:
                    mp3_bytes, tts_error = st.session_state[audio_key]
                    with audio_slot.container():
                        if mp3_bytes:
                            st.success("✅ Audio ready! Press play or download below.", icon="🎵")
                            st.audio(mp3_bytes, format="audio/mp3")
                            st.download_button(
                                "⬇ Download Audio (.mp3)",
                                data=mp3_bytes,
                                file_name=f"audio_{caption_lang.lower()}.mp3",
                                mime="audio/mpeg",
                            )
                        else:
                            st.error(f"Audio generation failed: {tts_error}", icon="❌")

                # Button always stays in the same position
                if st.button(f"🎙️ Generate {caption_lang} Audio", use_container_width=False):
                    st.session_state.pop(audio_key, None)
                    with audio_slot.container():
                        with st.spinner(f"Generating {caption_lang} audio via Google TTS…"):
                            mp3_bytes, tts_error = _text_to_mp3(translated_captions, caption_lang)
                        st.session_state[audio_key] = (mp3_bytes, tts_error)
                        if mp3_bytes:
                            st.success("✅ Audio ready! Press play or download below.", icon="🎵")
                            st.audio(mp3_bytes, format="audio/mp3")
                            st.download_button(
                                "⬇ Download Audio (.mp3)",
                                data=mp3_bytes,
                                file_name=f"audio_{caption_lang.lower()}.mp3",
                                mime="audio/mpeg",
                            )
                        else:
                            st.error(f"Audio generation failed: {tts_error}", icon="❌")
