#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local (ffmpeg) concatenation utility.

Use case: DashScope LLM may be unavailable (e.g., Arrearage), but you still want to
concatenate generated clips and add background music.

Examples (PowerShell):
  python examples/local_concat_with_music.py `
    --video "<url_or_path_1>" --video "<url_or_path_2>" `
    --music "C:\\path\\to\\bgm.mp3" --music-seconds 10
"""

import argparse
import os
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlparse, unquote_plus


def _is_http(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def _is_file_url(u: str) -> bool:
    return isinstance(u, str) and u.startswith("file://")


def _file_url_to_local_path(u: str) -> str:
    p = urlparse(u)
    if p.netloc:
        return p.netloc + unquote_plus(p.path)
    return unquote_plus(p.path)


def _download(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "ffmpeg failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True, help="Video URL/path. Repeat for multiple clips.")
    ap.add_argument("--music", required=True, help="Music mp3 URL/path/file://.")
    ap.add_argument("--music-seconds", type=float, default=10.0, help="Use only the first N seconds of music.")
    ap.add_argument(
        "--out",
        default="",
        help="Output mp4 path (default: workspace/edits/concat_<timestamp>.mp4).",
    )
    args = ap.parse_args()

    videos = [v.strip() for v in (args.video or []) if v and v.strip()]
    if len(videos) < 2:
        raise SystemExit("Need at least 2 --video inputs")

    tmp_dir = tempfile.mkdtemp(prefix="local_edit_")
    parts: list[str] = []

    # Stage videos locally for concat demuxer
    for i, v in enumerate(videos):
        if _is_file_url(v):
            v = _file_url_to_local_path(v)
        if _is_http(v):
            dst = os.path.join(tmp_dir, f"part_{i}.mp4")
            print(f"Downloading video {i+1}: {v}")
            _download(v, dst)
            parts.append(dst)
        else:
            if not os.path.exists(v):
                raise SystemExit(f"Video not found: {v}")
            parts.append(os.path.abspath(v))

    list_file = os.path.join(tmp_dir, "inputs.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            safe = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    concat_path = os.path.join(tmp_dir, "concat.mp4")
    print("Concatenating...")
    try:
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", concat_path])
    except Exception as e:
        # fallback re-encode if stream copy fails
        print(f"Concat stream copy failed, fallback to re-encode: {e}")
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                concat_path,
            ]
        )

    music = args.music.strip()
    if _is_file_url(music):
        music = _file_url_to_local_path(music)
    if _is_http(music):
        mpath = os.path.join(tmp_dir, "music.mp3")
        print(f"Downloading music: {music}")
        _download(music, mpath)
        music = mpath
    else:
        if not os.path.exists(music):
            raise SystemExit(f"Music not found: {music}")
        music = os.path.abspath(music)

    out = args.out.strip()
    if not out:
        os.makedirs(os.path.join("workspace", "edits"), exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join("workspace", "edits", f"concat_{ts}.mp4")
    out = os.path.abspath(out)

    # Mix: keep video, add trimmed music as the only audio track (first N seconds).
    # If video is longer than N seconds, audio will end early (silence afterwards).
    ms = max(0.0, float(args.music_seconds or 0.0))
    print(f"Adding music (first {ms}s)...")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            concat_path,
            "-i",
            music,
            "-filter_complex",
            f"[1:a]atrim=0:{ms},asetpts=N/SR/TB[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            out,
        ]
    )

    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
