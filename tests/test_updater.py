import hashlib
import os
import shutil
import tempfile
import unittest
from unittest import mock

import updater_github as u


class UpdaterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_upd_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(u._safe_remove, self.tmp_exe_path())

    def tmp_exe_path(self):
        return os.path.join(tempfile.gettempdir(), "gui_classifier_1.1.0.exe")

    def fake_release(self, tag="v1.1.0", with_digest=True):
        assets = [
            {"name": "御3T分图工具.exe", "browser_download_url": "http://example/exe"},
        ]
        if with_digest:
            assets.append({"name": "御3T分图工具.exe.sha256", "browser_download_url": "http://example/exe.sha256"})
        return {"tag_name": tag, "body": "发布说明", "assets": assets}

    def fake_download(self, url, dest):
        if url.endswith(".sha256"):
            with open(dest, "w", encoding="utf-8") as f:
                f.write("0" * 64)
        else:
            with open(dest, "wb") as f:
                f.write(b"fake-exe-bytes")

    def test_parse_ver(self):
        self.assertEqual(u._parse_ver("v1.2.10"), (1, 2, 10))
        self.assertEqual(u._parse_ver("1.2.3"), (1, 2, 3))
        self.assertTrue(u._parse_ver("v1.2.10") > u._parse_ver("1.2.3"))

    def test_has_update(self):
        self.assertTrue(u.has_update("1.0.1", "v1.1.0"))
        self.assertFalse(u.has_update("1.1.0", "v1.1.0"))
        self.assertFalse(u.has_update("1.2.0", "v1.1.0"))

    def test_select_asset(self):
        self.assertEqual(u.select_asset(self.fake_release())["name"], "御3T分图工具.exe")

    def test_find_digest_asset(self):
        release = self.fake_release()
        self.assertIsNotNone(u.find_digest_asset(release, "御3T分图工具.exe"))
        self.assertIsNone(u.find_digest_asset(self.fake_release(with_digest=False), "御3T分图工具.exe"))

    def test_extract_sha256(self):
        self.assertEqual(u._extract_sha256("hash = " + "a" * 64), "a" * 64)
        self.assertIsNone(u._extract_sha256("no hash here"))

    def test_sha256sum(self):
        p = os.path.join(self.tmp, "a.bin")
        with open(p, "wb") as f:
            f.write(b"abc")
        self.assertEqual(u.sha256sum(p), hashlib.sha256(b"abc").hexdigest())

    def test_prepare_update_sha_mismatch_removes_file(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release()):
            with mock.patch.object(u, "download_file", side_effect=self.fake_download):
                with self.assertRaises(RuntimeError):
                    u.prepare_update("1.0.1", verify_sha256="f" * 64)
        self.assertFalse(os.path.exists(exe))

    def test_prepare_update_sha_match(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        expected = hashlib.sha256(b"fake-exe-bytes").hexdigest()
        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release()):
            with mock.patch.object(u, "download_file", side_effect=self.fake_download):
                tag, notes, path = u.prepare_update("1.0.1", verify_sha256=expected)
        self.assertEqual(tag, "v1.1.0")
        self.assertEqual(notes, "发布说明")
        self.assertTrue(os.path.isfile(path))

    def test_prepare_update_missing_digest_asset_cancels(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release(with_digest=False)):
            with mock.patch.object(u, "download_file", side_effect=self.fake_download):
                with self.assertRaises(RuntimeError):
                    u.prepare_update("1.0.1")
        self.assertFalse(os.path.exists(exe))

    def test_prepare_update_digest_mismatch_cancels(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release()):
            with mock.patch.object(u, "download_file", side_effect=self.fake_download):
                with self.assertRaises(RuntimeError):
                    u.prepare_update("1.0.1")
        self.assertFalse(os.path.exists(exe))

    def test_prepare_update_digest_match(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        digest = hashlib.sha256(b"fake-exe-bytes").hexdigest()

        def download(url, dest):
            if url.endswith(".sha256"):
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(digest)
            else:
                with open(dest, "wb") as f:
                    f.write(b"fake-exe-bytes")

        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release()):
            with mock.patch.object(u, "download_file", side_effect=download):
                _tag, _notes, path = u.prepare_update("1.0.1")
        self.assertTrue(os.path.isfile(path))

    def test_prepare_update_no_verify_mode(self):
        exe = self.tmp_exe_path()
        u._safe_remove(exe)
        with mock.patch.object(u, "fetch_latest_release", return_value=self.fake_release(with_digest=False)):
            with mock.patch.object(u, "download_file", side_effect=self.fake_download):
                _tag, _notes, path = u.prepare_update("1.0.1", require_verify=False)
        self.assertTrue(os.path.isfile(path))

    def test_apply_update_invalid_file(self):
        with self.assertRaises(RuntimeError):
            u.apply_update_and_restart(os.path.join(self.tmp, "missing.exe"))

    def test_create_bat(self):
        bat = u.create_apply_update_bat("C:/x/御3T分图工具.exe", "C:/tmp/new.exe")
        content = open(bat, encoding="utf-8").read()
        self.assertIn("copy /Y", content)
        self.assertIn("%BAK%", content)
        self.assertIn("exit /b 4", content)
        self.assertIn("C:/x/御3T分图工具.exe", content)
        self.assertIn("C:/x/御3T分图工具.exe.bak", content)

    def test_check_interval(self):
        u.mark_checked()
        self.assertFalse(u.should_check_now(min_interval_sec=3600))
        self.assertTrue(u.should_check_now(min_interval_sec=-1))


if __name__ == "__main__":
    unittest.main()
