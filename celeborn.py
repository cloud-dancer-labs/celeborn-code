"""Celeborn thin installer — fetch, verify, place, and run the Celeborn binary.

This module is the entire `celeborn-code` PyPI package (BUSL 1.1 — see LICENSE).
The Celeborn client itself ships as a compiled, all-rights-reserved binary
published at https://github.com/cloud-dancer-labs/celeborn-releases. On first
run this installer:

  1. resolves your platform (macOS arm64, Linux x86_64, Windows x86_64),
  2. downloads the version-pinned release tarball,
  3. verifies its sha256 against the checksum baked into this file at release
     time (the LICENSE Supplemental Terms make checksum-valid installs the
     boundary of warranty and support — this step is that boundary),
  4. places the binary under ~/.celeborn/bin/ and
  5. execs it, passing your arguments through.

Every later run skips straight to step 5 — this shim stays out of the way.

Diagnostics: `celeborn --installer-info` prints what the installer resolved
(platform, URL, binary path) without touching the network. Env overrides:
CELEBORN_BIN_DIR relocates the binary directory; CELEBORN_INSTALLER_PLATFORM
forces the platform key (testing).

Questions: support chat at https://celeborncode.ai/faq
"""

import hashlib
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "0.3.0"

RELEASES = "https://github.com/cloud-dancer-labs/celeborn-releases/releases/download"

# sha256 of each platform tarball, from the release's SHA256SUMS manifest.
# Stamped per release — a platform absent here has no published binary for
# this version (the installer refuses honestly rather than skip verification).
CHECKSUMS = {
    "linux-x86_64": "187a121cc458dc1f508f3644fcae0f49aaf0842f6a0fc4066844e7d38475aff0",
    "macos-arm64": "185abf540e89267ce4152fd5bb37f142c57b0f11119f8329bdbe6560114e7140",
    "windows-x86_64": "8b4ffa9a7e580b4aacbd4077988a1d7f2b50775d82de67954413f2f9d97de4b2",
}

SUPPORT = "https://celeborncode.ai/faq"


def platform_key():
    """Map this machine to a release-asset platform key (may be unpublished)."""
    forced = os.environ.get("CELEBORN_INSTALLER_PLATFORM")
    if forced:
        return forced
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-x86_64"
    if sys.platform.startswith("linux"):
        return "linux-x86_64" if machine in ("x86_64", "amd64") else "linux-" + machine
    if sys.platform == "win32":
        return "windows-x86_64" if machine in ("amd64", "x86_64") else "windows-" + machine
    return sys.platform + "-" + (machine or "unknown")


def bin_dir():
    override = os.environ.get("CELEBORN_BIN_DIR")
    return Path(override) if override else Path.home() / ".celeborn" / "bin"


def binary_name():
    return "celeborn-%s%s" % (VERSION, ".exe" if sys.platform == "win32" else "")


def tarball_name(plat):
    return "celeborn-%s-%s.tar.gz" % (VERSION, plat)


def download_url(plat):
    return "%s/v%s/%s" % (RELEASES, VERSION, tarball_name(plat))


def _say(msg):
    sys.stderr.write("celeborn installer: %s\n" % msg)


def _fail(msg):
    _say(msg)
    _say("support chat: %s" % SUPPORT)
    return 1


def _download_verified(url, expected_sha, dest_dir):
    """Stream url to a temp file in dest_dir, verifying sha256. Returns Path or None."""
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "celeborn-installer/" + VERSION})
    fd, tmp_name = tempfile.mkstemp(prefix=".celeborn-dl-", dir=str(dest_dir))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        _say("DOWNLOAD FAILED sha256 verification for %s" % url)
        _say("  expected %s" % expected_sha)
        _say("  got      %s" % digest.hexdigest())
        _say("Refusing to install. Per the LICENSE Supplemental Terms, warranty and")
        _say("support cover only checksum-valid installs — a mismatched download is")
        _say("never placed. Re-run to retry; if it persists, contact support.")
        return None
    return tmp


def _extract_binary(tarball, plat, dest):
    """Pull just the binary (and its LICENSE) out of the verified tarball into dest's dir."""
    top = "celeborn-%s-%s" % (VERSION, plat)
    member = "%s/celeborn%s" % (top, ".exe" if plat.startswith("windows") else "")
    with tarfile.open(str(tarball), "r:gz") as tar:
        src = tar.extractfile(member)
        if src is None:
            raise KeyError(member)
        fd, tmp_name = tempfile.mkstemp(prefix=".celeborn-bin-", dir=str(dest.parent))
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.chmod(tmp_name, 0o755)
        os.replace(tmp_name, str(dest))
        # Best-effort: keep the binary's own license (all rights reserved) beside it.
        try:
            lic = tar.extractfile("%s/LICENSE" % top)
            if lic is not None:
                (dest.parent / ("LICENSE-%s.txt" % VERSION)).write_bytes(lic.read())
        except Exception:
            pass


def _prune_old(home, keep):
    """Best-effort removal of superseded binaries (a running exe may resist on Windows)."""
    for p in home.glob("celeborn-*"):
        if p.name.startswith("celeborn-" + VERSION):
            continue
        if p != keep:
            try:
                p.unlink()
            except OSError:
                pass


def ensure_binary():
    """Return the path to a verified binary, downloading it if absent. Exits on failure."""
    plat = platform_key()
    home = bin_dir()
    exe = home / binary_name()
    if exe.exists():
        return exe
    expected = CHECKSUMS.get(plat)
    if expected is None:
        raise SystemExit(_fail(
            "no Celeborn %s binary is published for your platform (%s) yet.\n"
            "  published platforms: %s" % (VERSION, plat, ", ".join(sorted(CHECKSUMS)))
        ))
    home.mkdir(parents=True, exist_ok=True)
    url = download_url(plat)
    size_mb = {"linux-x86_64": 23, "macos-arm64": 10, "windows-x86_64": 11}.get(plat)
    _say("first run — fetching the Celeborn %s binary (%s%s)" % (
        VERSION, plat, ", ~%d MB" % size_mb if size_mb else ""))
    _say("  %s" % url)
    try:
        tarball = _download_verified(url, expected, home)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(_fail("download failed (%s). Check connectivity and re-run." % exc))
    if tarball is None:
        raise SystemExit(1)
    try:
        _extract_binary(tarball, plat, exe)
    except KeyError as missing:
        raise SystemExit(_fail("release tarball is malformed (missing %s)." % missing))
    finally:
        tarball.unlink(missing_ok=True)
    _prune_old(home, exe)
    _say("sha256 verified ✓ — installed to %s" % exe)
    return exe


def installer_info():
    plat = platform_key()
    exe = bin_dir() / binary_name()
    print("celeborn-code installer %s (BUSL 1.1)" % VERSION)
    print("  platform : %s%s" % (plat, "" if plat in CHECKSUMS else "  (no published binary)"))
    print("  source   : %s" % download_url(plat))
    print("  binary   : %s  (%s)" % (exe, "present" if exe.exists() else "not yet downloaded"))
    print("  support  : %s" % SUPPORT)
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--installer-info":
        return installer_info()
    exe = ensure_binary()
    argv = [str(exe)] + sys.argv[1:]
    if sys.platform == "win32":
        try:
            return subprocess.run(argv).returncode
        except KeyboardInterrupt:
            return 130
    os.execv(str(exe), argv)  # replaces this process; never returns


if __name__ == "__main__":
    raise SystemExit(main())
