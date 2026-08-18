import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request

from dify_kml_client import DifyApiError, DifyKmlClient
from kml_ledger_converter import convert_kml_bytes


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class DifyClientTests(unittest.TestCase):
    def test_run_workflow_uses_single_file_and_same_user(self):
        requests = []

        def opener(request: Request, timeout: float):
            requests.append(request)
            return FakeResponse(json.dumps({"data": {"status": "succeeded", "outputs": {}}}).encode())

        client = DifyKmlClient("https://ylpf100.top/v1", "secret", "device-1", opener=opener)
        client.run_workflow("file-123")
        body = json.loads(requests[0].data.decode())
        self.assertEqual(body["user"], "device-1")
        self.assertEqual(body["inputs"]["kml_file"]["upload_file_id"], "file-123")
        self.assertEqual(body["inputs"]["kml_file"]["type"], "custom")
        self.assertNotIn("secret", requests[0].data.decode())

    def test_missing_output_file_is_actionable(self):
        with self.assertRaises(DifyApiError) as context:
            DifyKmlClient._find_output_file({"result": "ok"})
        self.assertEqual(context.exception.code, "OUTPUT_FILE_MISSING")

    def test_generate_ledger_uploads_runs_downloads_and_validates(self):
        source = b"<?xml version=\"1.0\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Placemark><name>#1</name><Point><coordinates>113.25,23.5</coordinates></Point></Placemark></kml>"
        expected = convert_kml_bytes(source, "220kV测试线路A.kml")
        responses = [
            json.dumps({"id": "upload-1"}).encode(),
            json.dumps({"data": {"id": "run-1", "status": "succeeded", "outputs": {"file_name": expected.file_name, "line_name": "测试线路A", "tower_count": 1, "files": [{"url": "/files/tools/result.xlsx"}]}}}).encode(),
            expected.xlsx_bytes,
        ]

        def opener(request: Request, timeout: float):
            return FakeResponse(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            kml_path = Path(directory) / "测试线路A.kml"
            kml_path.write_bytes(source)
            client = DifyKmlClient("https://ylpf100.top/v1", "secret", "device-1", opener=opener)
            result = client.generate_ledger(kml_path)
        self.assertEqual(result.file_name, expected.file_name)
        self.assertEqual(result.tower_count, 1)
        self.assertEqual(result.workflow_run_id, "run-1")

    def test_run_workflow_includes_manual_overrides_only_when_present(self):
        requests = []

        def opener(request: Request, timeout: float):
            requests.append(request)
            return FakeResponse(json.dumps({"data": {"status": "succeeded", "outputs": {}}}).encode())

        client = DifyKmlClient("https://ylpf100.top/v1", "secret", "device-1", opener=opener)
        client.run_workflow(
            "file-123",
            overrides={"voltage_level": "220", "line_name_1": "甲线", "line_name_2": "乙线", "circuit_type": "双回"},
        )
        body = json.loads(requests[0].data.decode())
        self.assertEqual(body["inputs"]["voltage_level"], "220")
        self.assertEqual(body["inputs"]["line_name_1"], "甲线")
        self.assertEqual(body["inputs"]["line_name_2"], "乙线")

    def test_generate_ledgers_supports_multiple_files(self):
        source = b"<?xml version=\"1.0\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Placemark><name>N1</name><Point><coordinates>113.25,23.5</coordinates></Point></Placemark></kml>"
        expected_a = convert_kml_bytes(source, "220kV甲线.kml")
        expected_b = convert_kml_bytes(source, "220kV乙线.kml")
        responses = [
            json.dumps({"id": "upload-1"}).encode(),
            json.dumps({
                "data": {
                    "id": "run-2",
                    "status": "succeeded",
                    "outputs": {
                        "status": "succeeded",
                        "line_names": ["甲线", "乙线"],
                        "file_names": ["甲线经纬度台账.xlsx", "乙线经纬度台账.xlsx"],
                        "files": [{"url": "/files/a.xlsx", "name": "甲线经纬度台账.xlsx"}, {"url": "/files/b.xlsx", "name": "乙线经纬度台账.xlsx"}],
                    },
                }
            }).encode(),
            expected_a.xlsx_bytes,
            expected_b.xlsx_bytes,
        ]

        def opener(request: Request, timeout: float):
            return FakeResponse(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            kml_path = Path(directory) / "220kV甲乙线N1-N1.kml"
            kml_path.write_bytes(source)
            client = DifyKmlClient("https://ylpf100.top/v1", "secret", "device-1", opener=opener)
            result = client.generate_ledgers(kml_path)
        self.assertFalse(result.manual_required)
        self.assertEqual([item.line_name for item in result.artifacts], ["甲线", "乙线"])
        self.assertEqual(result.workflow_run_id, "run-2")

    def test_generate_ledgers_returns_manual_required_without_download(self):
        source = b"<?xml version=\"1.0\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Placemark><name>N1</name><Point><coordinates>113.25,23.5</coordinates></Point></Placemark></kml>"
        responses = [
            json.dumps({"id": "upload-1"}).encode(),
            json.dumps({"data": {"id": "run-3", "status": "succeeded", "outputs": {"status": "manual_required", "manual_required": True, "manual_reason": "请补充电压等级", "line_names": ["异常线"]}}}).encode(),
        ]

        def opener(request: Request, timeout: float):
            return FakeResponse(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            kml_path = Path(directory) / "异常线.kml"
            kml_path.write_bytes(source)
            client = DifyKmlClient("https://ylpf100.top/v1", "secret", "device-1", opener=opener)
            result = client.generate_ledgers(kml_path)
        self.assertTrue(result.manual_required)
        self.assertIn("电压", result.manual_reason)
        self.assertEqual(result.upload_file_id, "upload-1")
