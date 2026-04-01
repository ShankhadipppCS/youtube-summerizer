"""
utils.py — helper functions for YT Summarizer (Advanced Edition)
Compatible with Python 3.11+  |  youtube-transcript-api v1.x
AI backend : Llama 3.3-70B  via Groq Cloud API
STT backend: Whisper large-v3 via Groq Cloud API (fallback when captions unavailable)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)


# ── Browser-like headers ──────────────────────────────────────────────────────

_YT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Referer": "https://www.youtube.com/",
}


# ── Video ID extraction ───────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """
    Handles all YouTube URL formats including ?si= tracking params:
      youtu.be/ID?si=xxx  |  youtube.com/watch?v=ID&si=xxx
      youtube.com/shorts/ID  |  youtube.com/embed/ID
    """
    url = url.strip()
    parsed = urlparse(url)

    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if _valid_id(vid) else None

    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            vid = qs["v"][0]
            return vid if _valid_id(vid) else None
        match = re.search(r"(?:embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if match:
            return match.group(1)

    match = re.fullmatch(r"[A-Za-z0-9_-]{11}", url)
    if match:
        return url
    return None


def _valid_id(vid: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", vid))


# ── Duration / views helpers ──────────────────────────────────────────────────

def _parse_iso_duration(iso: str) -> str:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return iso
    h, mn, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"


def _fmt_views(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── Video metadata ────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict[str, Any] | None:
    return _scrape_metadata(video_id) or _oembed_metadata(video_id)


def _scrape_metadata(video_id: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers=_YT_HEADERS,
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        html = resp.text

        title = "Unknown"
        og_title = re.search(r'<meta property="og:title"\s+content="([^"]+)"', html)
        if og_title:
            title = og_title.group(1)
        else:
            t2 = re.search(r'"title"\s*:\s*\{"runs":\[{"text":"([^"]+)"', html)
            if t2:
                title = t2.group(1)

        author = "Unknown"
        a1 = re.search(r'"ownerChannelName"\s*:\s*"([^"]+)"', html)
        if a1:
            author = a1.group(1)
        else:
            a2 = re.search(r'"channelName"\s*:\s*"([^"]+)"', html)
            if a2:
                author = a2.group(1)
            else:
                a3 = re.search(r'"author"\s*:\s*"([^"]+)"', html)
                if a3:
                    author = a3.group(1)

        duration = "N/A"
        d1 = re.search(r'"lengthSeconds"\s*:\s*"(\d+)"', html)
        if d1:
            secs = int(d1.group(1))
            h, r = divmod(secs, 3600)
            mn, s = divmod(r, 60)
            duration = f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"
        else:
            d2 = re.search(r'"duration"\s*:\s*"(PT[^"]+)"', html)
            if d2:
                duration = _parse_iso_duration(d2.group(1))

        views = "N/A"
        v1 = re.search(r'"viewCount"\s*:\s*"(\d+)"', html)
        if v1:
            views = _fmt_views(int(v1.group(1)))
        else:
            v2 = re.search(r'"videoViewCountRenderer".*?"simpleText"\s*:\s*"([^"]+)"', html)
            if v2:
                views = v2.group(1)

        thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        th = re.search(r'<meta property="og:image"\s+content="([^"]+)"', html)
        if th:
            thumbnail = th.group(1)

        if title == "Unknown" and author == "Unknown":
            return None

        return {
            "title":     title,
            "author":    author,
            "thumbnail": thumbnail,
            "duration":  duration,
            "views":     views,
        }
    except Exception:
        return None


def _oembed_metadata(video_id: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title":     data.get("title", "Unknown"),
                "author":    data.get("author_name", "Unknown"),
                "thumbnail": data.get(
                    "thumbnail_url",
                    f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                ),
                "duration":  "N/A",
                "views":     "N/A",
            }
    except Exception:
        pass
    return None


# ── Cookie / session helpers ──────────────────────────────────────────────────

def _parse_netscape_cookies(cookies_txt: str) -> list[dict]:
    cookies: list[dict] = []
    for line in cookies_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, _, path, secure, _, name, value = parts[:7]
        cookies.append(
            {
                "domain": domain,
                "path":   path,
                "secure": secure.upper() == "TRUE",
                "name":   name,
                "value":  value,
            }
        )
    return cookies


def _session_from_cookies(cookies_txt: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update(_YT_HEADERS)
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    session.cookies.set(
        "SOCS",
        "CAESEwgDEgk0OTc5NTkzNzIaAmVuIAEaBgiAo_CmBg",
        domain=".youtube.com",
    )
    if cookies_txt:
        for ck in _parse_netscape_cookies(cookies_txt):
            session.cookies.set(
                ck["name"],
                ck["value"],
                domain=ck["domain"],
                path=ck["path"],
            )
    return session


# ── Transcript fetching (YouTube captions) ────────────────────────────────────

def fetch_transcript(
    video_id: str,
    include_timestamps: bool = False,
    cookies_txt: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> tuple[str, str | None]:
    """
    Fetch transcript using youtube-transcript-api v1.x.
    Returns (text, error_message). error_message is None on success.
    """
    from youtube_transcript_api.proxies import WebshareProxyConfig

    def _best_transcript(tlist):
        # Prefer manually created transcripts in any language, then auto-generated
        try:
            return tlist.find_manually_created_transcript(["bn", "en", "en-US", "en-GB"])
        except NoTranscriptFound:
            pass
        try:
            return tlist.find_generated_transcript(["bn", "en", "en-US", "en-GB"])
        except NoTranscriptFound:
            pass
        # Fall back to any available transcript (manual first, then auto)
        for t in tlist:
            if not t.is_generated:
                return t
        return next(iter(tlist))

    def _run(ytt: YouTubeTranscriptApi) -> tuple[str, str | None]:
        try:
            tlist      = ytt.list(video_id)
            transcript = _best_transcript(tlist)
            segs       = transcript.fetch()
            text       = _segs_to_text(segs, include_timestamps)
            return text, None
        except TranscriptsDisabled:
            return "", "Transcripts are disabled for this video."
        except VideoUnavailable:
            return "", "Video is unavailable or private."
        except NoTranscriptFound:
            return "", "No transcript found — video may not have captions."
        except (RequestBlocked, CouldNotRetrieveTranscript):
            return "", "__blocked__"
        except Exception as exc:
            return "", f"Unexpected error: {exc}"

    if proxy_user and proxy_pass:
        try:
            proxy_cfg = WebshareProxyConfig(
                proxy_username=proxy_user,
                proxy_password=proxy_pass,
            )
            text, err = _run(YouTubeTranscriptApi(proxy_config=proxy_cfg))
            if text or (err and err != "__blocked__"):
                return text, err
        except Exception:
            pass

    session = _session_from_cookies(cookies_txt)
    text, err = _run(YouTubeTranscriptApi(http_client=session))
    if text or (err and err != "__blocked__"):
        return text, err

    time.sleep(1.5)
    text, err = _run(YouTubeTranscriptApi())
    if text or (err and err != "__blocked__"):
        return text, err

    if proxy_user and proxy_pass:
        return "", (
            "All fetch attempts failed (including proxy). "
            "Your proxy free quota may be exhausted."
        )
    return "", (
        "YouTube is rate-limiting transcript requests from this IP. "
        "Wait a minute and try again, or upload `cookies.txt` in the sidebar."
    )


def _segs_to_text(segments: Any, include_timestamps: bool) -> str:
    if include_timestamps:
        return "\n".join(f"[{_fmt_ts(seg.start)}] {seg.text}" for seg in segments)
    return " ".join(seg.text for seg in segments)


def _fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, r    = divmod(seconds, 3600)
    m, s    = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── yt-dlp audio download ─────────────────────────────────────────────────────

def _check_ytdlp() -> bool:
    """Return True if yt-dlp is available on PATH."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_audio(
    video_id: str, output_dir: str
) -> tuple[str | None, str | None]:
    """
    Download audio (mp3) from YouTube using yt-dlp.
    Returns (audio_file_path, error_message).
    Caps file size at 24MB to stay under Groq Whisper's 25MB limit.
    """
    if not _check_ytdlp():
        return None, (
            "yt-dlp is not installed. Install it with:\n"
            "  pip install yt-dlp\n"
            "Then restart the app."
        )

    url          = f"https://www.youtube.com/watch?v={video_id}"
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",   "mp3",
        "--audio-quality",  "5",      # medium quality — keeps file small
        "--max-filesize",   "24M",    # Groq Whisper hard limit is 25MB
        "--output",         out_template,
        "--no-progress",
        "--quiet",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Private video" in stderr or "private" in stderr.lower():
                return None, "Video is private — cannot download audio."
            if "not available" in stderr.lower():
                return None, "Video is not available in your region."
            if "File is larger than max-filesize" in stderr:
                return None, (
                    "Audio exceeds 24MB (Groq Whisper limit). "
                    "The video is likely very long (>2 hrs). Try a shorter clip."
                )
            return None, f"yt-dlp error: {stderr[:400]}"

        for f in Path(output_dir).glob(f"{video_id}.*"):
            return str(f), None

        return None, "Audio file not found after download."
    except subprocess.TimeoutExpired:
        return None, "Audio download timed out (>5 min)."
    except Exception as exc:
        return None, f"Download error: {exc}"


# ── Groq Whisper STT ──────────────────────────────────────────────────────────

_GROQ_WHISPER_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_WHISPER_MODEL    = "whisper-large-v3"

# ISO codes Whisper supports
WHISPER_LANGUAGES: dict[str, str] = {
    "Auto-detect": "",
    "Bengali":     "bn",
    "English":     "en",
    "Hindi":       "hi",
    "Spanish":     "es",
    "French":      "fr",
    "German":      "de",
    "Japanese":    "ja",
    "Arabic":      "ar",
    "Portuguese":  "pt",
    "Russian":     "ru",
    "Chinese":     "zh",
}


def transcribe_with_whisper(
    audio_path: str,
    api_key: str,
    language: str = "",           # ISO code; "" = auto-detect
) -> tuple[dict | None, str | None]:
    """
    Transcribe audio via Groq Whisper large-v3.

    Returns (result, error).
    result keys:
      "text"     — full plain transcript
      "segments" — list of {id, start, end, text}
      "language" — detected/requested language code
    """
    file_size = os.path.getsize(audio_path)
    if file_size > 25 * 1024 * 1024:
        return None, f"Audio is {file_size/1e6:.1f}MB — Groq limit is 25MB."

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with open(audio_path, "rb") as f:
            for attempt in range(3):
                f.seek(0)
                files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                data: dict[str, Any] = {
                    "model":           _GROQ_WHISPER_MODEL,
                    "response_format": "verbose_json",
                    "temperature":     "0",
                }
                if language:
                    data["language"] = language

                resp = requests.post(
                    _GROQ_WHISPER_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 30 * (attempt + 1)))
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    return None, f"Groq Whisper error {resp.status_code}: {resp.text[:300]}"

                raw      = resp.json()
                segments = [
                    {
                        "id":    seg.get("id", i),
                        "start": float(seg.get("start", 0)),
                        "end":   float(seg.get("end",   0)),
                        "text":  seg.get("text", "").strip(),
                    }
                    for i, seg in enumerate(raw.get("segments", []))
                ]
                return {
                    "text":     raw.get("text", "").strip(),
                    "segments": segments,
                    "language": raw.get("language", "unknown"),
                }, None

        return None, "Whisper failed after 3 retries (rate limit)."
    except Exception as exc:
        return None, f"Whisper error: {exc}"


# ── Combined fetch: YouTube captions → Whisper fallback ──────────────────────

def fetch_transcript_with_whisper_fallback(
    video_id: str,
    api_key: str,
    include_timestamps: bool = False,
    cookies_txt: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
    whisper_language: str = "",
) -> tuple[str, str | None, list[dict] | None, str]:
    """
    Try YouTube captions first; fall back to Groq Whisper if unavailable.

    Returns:
      (transcript_text, error_message, whisper_segments, source)
      whisper_segments — list of {id, start, end, text}, set only when Whisper ran
      source           — "youtube" | "whisper"
    """
    # Step 1 — try YouTube captions
    text, err = fetch_transcript(
        video_id,
        include_timestamps=include_timestamps,
        cookies_txt=cookies_txt,
        proxy_user=proxy_user,
        proxy_pass=proxy_pass,
    )

    if text:
        return text, None, None, "youtube"

    # Hard errors — no point trying Whisper
    hard_errors = {
        "Video is unavailable or private.",
        "Transcripts are disabled for this video.",
    }
    if err in hard_errors:
        return "", err, None, "youtube"

    # Step 2 — Whisper fallback
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path, dl_err = download_audio(video_id, tmp_dir)
        if dl_err:
            return "", f"No captions found & audio download failed:\n{dl_err}", None, "whisper"

        result, w_err = transcribe_with_whisper(audio_path, api_key, language=whisper_language)
        if w_err:
            return "", f"Whisper transcription failed: {w_err}", None, "whisper"

        plain_text = result["text"]
        if include_timestamps:
            plain_text = "\n".join(
                f"[{_fmt_ts(seg['start'])}] {seg['text']}"
                for seg in result["segments"]
            )

        return plain_text, None, result["segments"], "whisper"


# ── SRT helpers ───────────────────────────────────────────────────────────────

def _srt_ts(seconds: float) -> str:
    """Format as SRT timestamp: HH:MM:SS,mmm"""
    ms       = int(round((seconds % 1) * 1000))
    s        = int(seconds)
    h, rem   = divmod(s, 3600)
    m, s     = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Convert Whisper segment dicts to SRT format string."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if text:
            lines.append(
                f"{i}\n{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}\n{text}\n"
            )
    return "\n".join(lines)


def translate_segments_to_srt(
    segments: list[dict],
    target_language: str,
    api_key: str,
) -> str:
    """
    Translate Whisper segments to target_language via Llama (in batches),
    preserving timestamps, and return as SRT string.
    """
    BATCH = 40
    translated_segs: list[dict] = []

    for batch_start in range(0, len(segments), BATCH):
        batch    = segments[batch_start: batch_start + BATCH]
        numbered = "\n".join(
            f"{i + 1}. {seg['text'].strip()}"
            for i, seg in enumerate(batch)
        )
        system = (
            f"You are a professional subtitle translator. "
            f"Translate each numbered line to {target_language}. "
            "Reply ONLY with the same numbered list — no explanations, no markdown."
        )
        raw = _call_llama(system, numbered, api_key, max_tokens=2048, temperature=0.2)

        translated_lines: dict[int, str] = {}
        for line in raw.strip().splitlines():
            m = re.match(r"^(\d+)\.\s*(.*)", line.strip())
            if m:
                translated_lines[int(m.group(1))] = m.group(2).strip()

        for i, seg in enumerate(batch):
            translated_segs.append({
                **seg,
                "text": translated_lines.get(i + 1, seg["text"]),
            })

    return segments_to_srt(translated_segs)


def translate_transcript_to_language(
    transcript: str,
    target_language: str,
    api_key: str,
) -> str:
    """Translate a plain-text transcript to target_language via Llama."""
    if len(transcript) <= 20_000:
        system = (
            f"You are a professional translator. "
            f"Translate the following to {target_language}. "
            "Preserve line breaks. Output only the translation."
        )
        return _call_llama(system, transcript, api_key, max_tokens=2048, temperature=0.2)

    chunks = _chunk_transcript(transcript, chunk_size=18_000, overlap=200)
    parts: list[str] = []
    for chunk in chunks:
        system = (
            f"Translate to {target_language}. Preserve line breaks. Output only the translation."
        )
        parts.append(_call_llama(system, chunk, api_key, max_tokens=2048, temperature=0.2))
    return "\n".join(parts)


# ── Groq / Llama 3.3-70B ─────────────────────────────────────────────────────

_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL    = "llama-3.3-70b-versatile"

_STYLE_INSTRUCTIONS: dict[str, str] = {
    "Concise":         "Provide a concise summary in 3-5 short paragraphs.",
    "Detailed":        "Provide a thorough, detailed summary covering all major topics discussed.",
    "Bullet Points":   "Summarize using clearly organized bullet points grouped by topic.",
    "Executive Brief": (
        "Write an executive brief: one-line TL;DR, followed by 3-5 high-impact insights, "
        "and a recommended action or takeaway."
    ),
}


def _call_llama(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       _GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.9,
    }

    for attempt in range(3):
        resp = requests.post(_GROQ_ENDPOINT, headers=headers, json=payload, timeout=90)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 30 * (attempt + 1)))
            print(f"[Groq 429] attempt={attempt + 1}, waiting {retry_after}s")
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected Groq response: {data}") from exc

    raise RuntimeError("Groq rate limit hit after 3 retries. Please wait and try again.")


# ── Long-video chunked summarization ─────────────────────────────────────────

_CHUNK_SIZE    = 40_000   # larger chunks → fewer API calls for long videos
_CHUNK_OVERLAP = 200


def _chunk_transcript(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def summarize_with_llama(
    transcript: str,
    api_key: str,
    style: str = "Concise",
    language: str = "English",
    max_tokens: int = 1024,
) -> str:
    instr = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["Concise"])

    if len(transcript) <= _CHUNK_SIZE:
        system = (
            "You are an expert content analyst. "
            f"Summarize YouTube video transcripts. Style: {instr} "
            f"Always respond in {language}. No preamble."
        )
        return _call_llama(system, f"Transcript:\n{transcript}", api_key, max_tokens)

    # Map step
    chunks          = _chunk_transcript(transcript)
    chunk_summaries = []
    chunk_system    = (
        f"Summarize this single segment of a long YouTube video concisely in {language}."
    )
    for i, chunk in enumerate(chunks, 1):
        partial = _call_llama(
            chunk_system,
            f"Segment {i}/{len(chunks)}:\n{chunk}",
            api_key, max_tokens=300, temperature=0.3,
        )
        chunk_summaries.append(f"[Seg {i}/{len(chunks)}]\n{partial}")

    # Reduce step
    merge_system = (
        "Merge these per-segment summaries of a long YouTube video into one coherent summary. "
        f"Style: {instr} Respond in {language}. No preamble."
    )
    return _call_llama(
        merge_system,
        "Summaries:\n\n" + "\n\n".join(chunk_summaries),
        api_key, max_tokens, temperature=0.4,
    )


# Backward-compatible alias
summarize_with_gemini = summarize_with_llama


def generate_key_points(
    transcript: str, api_key: str, max_tokens: int = 1024, summary: str | None = None,
    language: str = "English",
) -> list[str] | str:
    system = (
        "You are an expert analyst. Extract key points from YouTube transcripts. "
        f"Always respond in {language}. "
        "Respond ONLY with a valid JSON array of strings — no markdown, no extra text."
    )
    # Reuse pre-computed summary if provided — avoids re-summarizing the full transcript
    source = summary if summary else (
        summarize_with_llama(transcript, api_key, style="Detailed", language=language, max_tokens=1500)
        if len(transcript) > _CHUNK_SIZE
        else transcript[:30_000]
    )
    raw = _call_llama(system, f"Extract 7 key points:\n{source}", api_key, max_tokens)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        pts   = json.loads(clean)
        if isinstance(pts, list):
            return [str(p) for p in pts]
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def generate_quiz(
    transcript: str, api_key: str, n_questions: int = 5, summary: str | None = None,
    language: str = "English",
) -> list[dict] | str:
    system = (
        "You are a quiz generator. "
        f"Always respond in {language}. "
        'Respond ONLY with a valid JSON array: [{"question":"...","answer":"...","explanation":"..."}] '
        "No markdown, no extra text, no preamble."
    )
    # Reuse pre-computed summary if provided — avoids re-summarizing the full transcript
    source = summary if summary else (
        summarize_with_llama(transcript, api_key, style="Detailed", language=language, max_tokens=1500)
        if len(transcript) > _CHUNK_SIZE
        else transcript[:25_000]
    )
    raw = _call_llama(system, f"Create {n_questions} questions:\n{source}", api_key, 1024)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        items = json.loads(clean)
        if isinstance(items, list):
            return items
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


# ── Export ────────────────────────────────────────────────────────────────────

def export_summary_as_txt(
    title: str, summary: str, key_points: list[str] | str
) -> str:
    sep   = "=" * 60
    lines = [sep, f"YouTube Video Summary: {title}", sep, "", "SUMMARY", "-------", summary,
             "", "KEY POINTS", "----------"]
    if isinstance(key_points, list):
        for i, pt in enumerate(key_points, 1):
            lines.append(f"{i}. {pt}")
    else:
        lines.append(key_points)
    lines += ["", sep]
    return "\n".join(lines)
