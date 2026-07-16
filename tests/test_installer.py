"""Offline tests for the thin installer (no network, no real ~/.celeborn).

Run: python3 -m unittest discover tests
"""

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import celeborn  # noqa: E402


def make_tarball(dest_dir, plat, binary_bytes=b"#!/bin/sh\necho fake-celeborn\n"):
    """Craft a release-shaped tarball: celeborn-<ver>-<plat>/{celeborn,LICENSE}."""
    top = "celeborn-%s-%s" % (celeborn.VERSION, plat)
    exe_name = "celeborn.exe" if plat.startswith("windows") else "celeborn"
    path = Path(dest_dir) / celeborn.tarball_name(plat)
    with tarfile.open(str(path), "w:gz") as tar:
        for name, data in ((exe_name, binary_bytes), ("LICENSE", b"all rights reserved\n")):
            info = tarfile.TarInfo("%s/%s" % (top, name))
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
    return path


class PlatformKey(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"CELEBORN_INSTALLER_PLATFORM": "linux-x86_64"}):
            self.assertEqual(celeborn.platform_key(), "linux-x86_64")

    def _key(self, plat, machine):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CELEBORN_INSTALLER_PLATFORM", None)
            with mock.patch.object(celeborn.sys, "platform", plat), \
                 mock.patch.object(celeborn.platform, "machine", return_value=machine):
                return celeborn.platform_key()

    def test_mappings(self):
        self.assertEqual(self._key("darwin", "arm64"), "macos-arm64")
        self.assertEqual(self._key("darwin", "x86_64"), "macos-x86_64")
        self.assertEqual(self._key("linux", "x86_64"), "linux-x86_64")
        self.assertEqual(self._key("linux", "AMD64"), "linux-x86_64")
        self.assertEqual(self._key("linux", "aarch64"), "linux-aarch64")
        self.assertEqual(self._key("win32", "AMD64"), "windows-x86_64")


class Checksums(unittest.TestCase):
    def test_table_shape(self):
        self.assertIn("macos-arm64", celeborn.CHECKSUMS)
        self.assertIn("linux-x86_64", celeborn.CHECKSUMS)
        self.assertIn("windows-x86_64", celeborn.CHECKSUMS)
        for plat, sha in celeborn.CHECKSUMS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", plat)

    def test_url_shape(self):
        self.assertEqual(
            celeborn.download_url("macos-arm64"),
            "https://github.com/cloud-dancer-labs/celeborn-releases/releases/download/"
            "v%s/celeborn-%s-macos-arm64.tar.gz" % (celeborn.VERSION, celeborn.VERSION),
        )


class EnsureBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "bin"
        self.env = mock.patch.dict(os.environ, {
            "CELEBORN_BIN_DIR": str(self.home),
            "CELEBORN_INSTALLER_PLATFORM": "linux-x86_64",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _serve(self, tarball):
        """Point the installer's download at a local file, with the right checksum."""
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        url_patch = mock.patch.object(
            celeborn, "download_url", lambda plat: tarball.resolve().as_uri())
        sums_patch = mock.patch.dict(celeborn.CHECKSUMS, {"linux-x86_64": sha})
        return url_patch, sums_patch

    def test_download_verify_place(self):
        with tempfile.TemporaryDirectory() as src:
            tarball = make_tarball(src, "linux-x86_64")
            url_patch, sums_patch = self._serve(tarball)
            with url_patch, sums_patch:
                exe = celeborn.ensure_binary()
        self.assertTrue(exe.exists())
        self.assertEqual(exe.name, "celeborn-" + celeborn.VERSION)
        self.assertTrue(os.access(str(exe), os.X_OK))
        # the binary's own license landed beside it
        self.assertTrue((self.home / ("LICENSE-%s.txt" % celeborn.VERSION)).exists())
        # second call: no download path taken at all
        with mock.patch.object(celeborn, "_download_verified") as dl:
            self.assertEqual(celeborn.ensure_binary(), exe)
            dl.assert_not_called()

    def test_checksum_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as src:
            tarball = make_tarball(src, "linux-x86_64")
            url_patch, _ = self._serve(tarball)
            bad = mock.patch.dict(celeborn.CHECKSUMS, {"linux-x86_64": "0" * 64})
            with url_patch, bad, self.assertRaises(SystemExit) as ctx:
                celeborn.ensure_binary()
        self.assertEqual(ctx.exception.code, 1)
        self.assertFalse((self.home / ("celeborn-" + celeborn.VERSION)).exists())
        # no half-written residue
        self.assertEqual(list(self.home.glob(".celeborn-*")), [])

    def test_unpublished_platform_refuses_honestly(self):
        with mock.patch.dict(os.environ, {"CELEBORN_INSTALLER_PLATFORM": "macos-x86_64"}):
            with self.assertRaises(SystemExit) as ctx:
                celeborn.ensure_binary()
        self.assertEqual(ctx.exception.code, 1)

    def test_malformed_tarball_refuses(self):
        with tempfile.TemporaryDirectory() as src:
            # right checksum, wrong platform layout inside → missing member
            tarball = make_tarball(src, "windows-x86_64")
            tarball = tarball.rename(Path(src) / celeborn.tarball_name("linux-x86_64"))
            url_patch, sums_patch = self._serve(tarball)
            with url_patch, sums_patch, self.assertRaises(SystemExit) as ctx:
                celeborn.ensure_binary()
        self.assertEqual(ctx.exception.code, 1)

    def test_prune_old_versions(self):
        self.home.mkdir(parents=True)
        stale = self.home / "celeborn-0.2.9"
        stale.write_bytes(b"old")
        with tempfile.TemporaryDirectory() as src:
            tarball = make_tarball(src, "linux-x86_64")
            url_patch, sums_patch = self._serve(tarball)
            with url_patch, sums_patch:
                celeborn.ensure_binary()
        self.assertFalse(stale.exists())


class InstallerInfo(unittest.TestCase):
    def test_info_is_offline_and_zero(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"CELEBORN_BIN_DIR": d}), \
                 mock.patch.object(celeborn.urllib.request, "urlopen",
                                   side_effect=AssertionError("network touched")), \
                 mock.patch.object(sys, "argv", ["celeborn", "--installer-info"]), \
                 mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                self.assertEqual(celeborn.main(), 0)
        self.assertIn(celeborn.VERSION, out.getvalue())


if __name__ == "__main__":
    unittest.main()
