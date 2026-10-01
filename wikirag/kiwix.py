"""Optional kiwix-serve: started on demand, used for clickable source links (and by the v1 pipeline)."""

import atexit
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

# kiwix-serve process started by this program, stopped at exit
_kiwix_process = None


def find_kiwix_binary() -> Optional[str]:
    """Find kiwix-serve binary in common locations"""
    locations = [
        Path.home() / ".local/bin/kiwix-serve",
        Path("/usr/local/bin/kiwix-serve"),
        Path("/usr/bin/kiwix-serve"),
    ]
    for location in locations:
        if location.exists():
            return str(location)
    try:
        result = subprocess.run(["which", "kiwix-serve"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def is_kiwix_running(port=8080) -> bool:
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def start_kiwix_server(port=8080, zim_path: Path = None) -> bool:
    """Start kiwix-serve for zim_path on 127.0.0.1 unless one is already running"""
    global _kiwix_process

    if is_kiwix_running(port):
        print(f"✓ Kiwix server already running at http://localhost:{port}")
        return True

    print("📚 Starting Kiwix server...")
    kiwix_bin = find_kiwix_binary()
    if not kiwix_bin:
        print("❌ kiwix-serve not found. Install with:")
        print("   wget https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64.tar.gz")
        print("   tar xzf kiwix-tools_linux-x86_64.tar.gz")
        print("   mv kiwix-tools_*/kiwix-serve ~/.local/bin/")
        return False

    if zim_path is None:
        print("❌ No Wikipedia ZIM file found. Download with:")
        print("   scripts/setup_full_offline_wikipedia.sh")
        print("   or pass --zim PATH / set WIKI_ZIM")
        return False

    try:
        print(f"   ZIM: {zim_path}")
        cmd = [kiwix_bin, "--port", str(port), "--address", "127.0.0.1", str(zim_path)]
        _kiwix_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          start_new_session=True)
        for _ in range(10):
            time.sleep(0.5)
            if is_kiwix_running(port):
                print(f"✓ Kiwix server started at http://localhost:{port}")
                return True
        print("⚠ Kiwix server started but not responding")
        return False
    except Exception as e:
        print(f"❌ Failed to start Kiwix server: {e}")
        return False


def _cleanup_kiwix():
    """Stop kiwix-serve if we started it"""
    if _kiwix_process:
        try:
            os.killpg(os.getpgid(_kiwix_process.pid), signal.SIGTERM)
            _kiwix_process.wait(timeout=5)
            print("\n✓ Stopped Kiwix server")
        except Exception:
            pass


atexit.register(_cleanup_kiwix)


class KiwixServer:
    """A kiwix-serve instance and the URL prefix of the book being served."""

    def __init__(self, url: str = "http://localhost:8080", zim_path: Optional[Path] = None):
        self.url = url.rstrip('/')
        self.zim_path = zim_path
        self.content_base: Optional[str] = None

    def connect(self, auto_start: bool) -> bool:
        """True if kiwix-serve answers, starting it first if allowed; discovers content_base"""
        try:
            response = requests.get(f"{self.url}/", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Kiwix server at {self.url}")
            connected = True
        except Exception:
            connected = False
            if auto_start:
                port = int(self.url.split(":")[-1]) if ":" in self.url.split("//")[-1] else 8080
                connected = start_kiwix_server(port, self.zim_path)
        if connected:
            # e.g. http://localhost:8080/content/wikipedia_en_all_nopic_2026-06
            self.content_base = self.discover_content_base()
        return connected

    def discover_content_base(self) -> Optional[str]:
        """
        Find the book URL prefix from the server's OPDS catalog, preferring the
        book that matches self.zim_path. Falls back to the ZIM file stem.
        """
        stem = self.zim_path.stem if self.zim_path else None
        try:
            response = requests.get(f"{self.url}/catalog/v2/entries", params={'count': 100}, timeout=5)
            response.raise_for_status()
            hrefs = re.findall(r'<link[^>]*type="text/html"[^>]*href="([^"]*/content/[^"]+)"', response.text)
        except Exception:
            hrefs = []
        if hrefs:
            matching = [h for h in hrefs if stem and h.rstrip('/').endswith(f"/{stem}")]
            wikipedia = [h for h in hrefs if '/wikipedia_' in h]
            href = (matching or wikipedia or hrefs)[0].rstrip('/')
            return href if href.startswith('http') else f"{self.url}{href}"
        if stem:
            return f"{self.url}/content/{stem}"
        return None

    def article_url(self, title: str) -> Optional[str]:
        """Direct URL for an article title (titles are case-sensitive, spaces become underscores)"""
        if not self.content_base:
            return None
        return f"{self.content_base}/{quote(title.replace(' ', '_'))}"

    def path_url(self, path: str, anchor: str = None) -> Optional[str]:
        """URL for a ZIM entry path, optionally pointing at a section"""
        if not self.content_base:
            return None
        url = f"{self.content_base}/{quote(path)}"
        return f"{url}#{quote(anchor, safe='()')}" if anchor else url
