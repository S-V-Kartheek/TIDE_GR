"""Robust file download (resumable across Colab sessions via a .part file)."""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request

from ..utils import log


def md5sum(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str, md5: str | None = None, retries: int = 5) -> str:
    if os.path.exists(dest):
        if md5 is None or md5sum(dest) == md5:
            return dest
        log(f"[download] md5 mismatch for {dest}; downloading again")
        os.remove(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                code = getattr(r, "status", 200)
                mode = "ab" if (have and code == 206) else "wb"
                total = r.headers.get("Content-Length")
                total = int(total) + (have if mode == "ab" else 0) if total else None
                done = have if mode == "ab" else 0
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if time.time() - last > 15:
                            pct = f"{100 * done / total:.1f}%" if total else f"{done / 1e6:.0f} MB"
                            log(f"[download] {os.path.basename(dest)} {pct}")
                            last = time.time()
            os.replace(part, dest)
            if md5 is not None and md5sum(dest) != md5:
                os.remove(dest)
                raise IOError("md5 mismatch after download")
            log(f"[download] done: {dest}")
            return dest
        except Exception as e:  # network hiccup: retry (resumes from .part)
            log(f"[download] attempt {attempt}/{retries} failed: {e!r}")
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not download {url}")
