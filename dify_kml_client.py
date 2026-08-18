"""Small blocking client for the Dify file + workflow APIs.

The module contains no GUI code.  It uploads one local KML, invokes the
published workflow and downloads the returned XLSX after validating it with
the same ledger contract used by the local converter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import uuid

from kml_ledger_converter import KmlConversionError, validate_ledger_bytes


class DifyApiError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class DifyLedgerResult:
    file_name: str
    line_name: str
    tower_count: int
    warnings: tuple[str, ...]
    xlsx_bytes: bytes
    sha256: Optional[str]
    workflow_run_id: Optional[str]


@dataclass(frozen=True)
class DifyLedgerArtifact:
    file_name: str
    line_name: str
    tower_count: int
    warnings: tuple[str, ...]
    xlsx_bytes: bytes
    sha256: Optional[str]


@dataclass(frozen=True)
class DifyLedgerBatchResult:
    status: str
    manual_required: bool
    manual_reason: str
    source_file_name: str
    line_full_name: str
    voltage_level: str
    circuit_type: str
    line_names: tuple[str, ...]
    warnings: tuple[str, ...]
    artifacts: tuple[DifyLedgerArtifact, ...]
    workflow_run_id: Optional[str]
    upload_file_id: Optional[str]


def _json_error(status: int, body: bytes) -> DifyApiError:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
        message = payload.get("message") or payload.get("error") or str(payload)
    except Exception:
        message = body.decode("utf-8", errors="replace")[:500]
    if status in (401, 403):
        code = "AUTH_FAILED"
    elif status == 429:
        code = "RATE_LIMITED"
    elif status >= 500:
        code = "SERVICE_UNAVAILABLE"
    else:
        code = "API_REQUEST_FAILED"
    return DifyApiError(code, f"Dify API 返回 HTTP {status}: {message}", status=status)


class DifyKmlClient:
    """Client for the Dify endpoints used by the KML ledger workflow."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        user: str,
        *,
        timeout: float = 120.0,
        retries: int = 2,
        opener: Callable[..., Any] = urlopen,
    ):
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("Dify API Base URL 必须以 http:// 或 https:// 开头")
        if not api_key.strip():
            raise ValueError("Dify API Key 不能为空")
        if not user.strip():
            raise ValueError("Dify user 不能为空")
        self.base_url = base_url
        self.api_key = api_key.strip()
        self.user = user.strip()
        self.timeout = timeout
        self.retries = max(0, retries)
        self._opener = opener

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, request: Request) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                body = exc.read()
                error = _json_error(exc.code, body)
                if error.code in {"AUTH_FAILED", "RATE_LIMITED", "API_REQUEST_FAILED"} or attempt >= self.retries:
                    raise error from exc
                time.sleep(0.5 * (2**attempt))
            except (URLError, TimeoutError, OSError) as exc:
                if attempt >= self.retries:
                    raise DifyApiError("SERVICE_UNAVAILABLE", f"无法连接 Dify 服务: {exc}") from exc
                time.sleep(0.5 * (2**attempt))
        raise DifyApiError("SERVICE_UNAVAILABLE", "Dify 请求未完成")

    def upload_file(self, file_path: str | os.PathLike[str]) -> str:
        path = Path(file_path)
        if path.suffix.lower() != ".kml":
            raise DifyApiError("INVALID_FILE_TYPE", "只允许上传 .kml 文件")
        if not path.is_file():
            raise DifyApiError("LOCAL_FILE_MISSING", f"KML 文件不存在: {path}")
        boundary = f"----CodexDify{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/vnd.google-earth.kml+xml"
        raw = path.read_bytes()
        parts = []
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"user\"\r\n\r\n{self.user}\r\n".encode("utf-8")
        )
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            + raw
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode("ascii"))
        request = Request(
            f"{self.base_url}/files/upload",
            data=b"".join(parts),
            headers={**self._headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            payload = json.loads(self._request(request).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DifyApiError("API_RESPONSE_INVALID", "文件上传接口返回了无效 JSON") from exc
        file_id = payload.get("id")
        if not file_id:
            raise DifyApiError("UPLOAD_FAILED", "文件上传成功响应缺少 id")
        return str(file_id)

    def run_workflow(
        self,
        upload_file_id: str,
        *,
        file_type: str = "custom",
        overrides: Optional[dict[str, object]] = None,
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "kml_file": {
                "type": file_type,
                "transfer_method": "local_file",
                "upload_file_id": upload_file_id,
            }
        }
        for key in ("voltage_level", "circuit_type", "line_name_1", "line_name_2"):
            value = (overrides or {}).get(key)
            if value not in (None, ""):
                inputs[key] = str(value).strip()
        body = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": self.user,
        }
        request = Request(
            f"{self.base_url}/workflows/run",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={**self._headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            payload = json.loads(self._request(request).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DifyApiError("API_RESPONSE_INVALID", "Workflow 接口返回了无效 JSON") from exc
        data = payload.get("data") or payload
        if data.get("status") not in (None, "succeeded"):
            raise DifyApiError("WORKFLOW_FAILED", data.get("error") or data.get("message") or "Workflow 执行失败")
        return payload

    @staticmethod
    def _find_output_files(outputs: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("files", "ledger_file", "file"):
            value = outputs.get(key)
            if isinstance(value, dict):
                value = [value]
            if isinstance(value, list):
                files = [
                    item for item in value
                    if isinstance(item, dict)
                    and (item.get("url") or item.get("source_url") or item.get("download_url"))
                ]
                if files:
                    return files
        raise DifyApiError("OUTPUT_FILE_MISSING", "Workflow 成功但没有返回可下载的台账文件")

    @staticmethod
    def _find_output_file(outputs: dict[str, Any]) -> dict[str, Any]:
        return DifyKmlClient._find_output_files(outputs)[0]

    def _download(self, url: str) -> bytes:
        resolved_url = urljoin(self.base_url + "/", url)
        request = Request(resolved_url, headers=self._headers, method="GET")
        return self._request(request)

    def generate_ledgers(
        self,
        kml_path: str | os.PathLike[str],
        *,
        overrides: Optional[dict[str, object]] = None,
        upload_file_id: Optional[str] = None,
    ) -> DifyLedgerBatchResult:
        path = Path(kml_path)
        upload_id = upload_file_id or self.upload_file(path)
        response = self.run_workflow(upload_id, overrides=overrides)
        data = response.get("data") or response
        outputs = data.get("outputs") or {}
        warnings = outputs.get("warnings") or []
        if isinstance(warnings, str):
            warnings = [warnings]
        manual_required = bool(outputs.get("manual_required")) or str(outputs.get("status") or "").lower() == "manual_required"
        line_names_value = outputs.get("line_names") or []
        if isinstance(line_names_value, str):
            line_names_value = [line_names_value]
        line_names = tuple(str(item) for item in line_names_value if str(item).strip())
        if manual_required:
            return DifyLedgerBatchResult(
                status="manual_required",
                manual_required=True,
                manual_reason=str(outputs.get("manual_reason") or "需要补充线路信息"),
                source_file_name=path.name,
                line_full_name=str(outputs.get("line_full_name") or ""),
                voltage_level=str(outputs.get("voltage_level") or ""),
                circuit_type=str(outputs.get("circuit_type") or ""),
                line_names=line_names,
                warnings=tuple(str(item) for item in warnings),
                artifacts=(),
                workflow_run_id=data.get("id") or response.get("workflow_run_id"),
                upload_file_id=upload_id,
            )

        file_payloads = self._find_output_files(outputs)
        file_names_value = outputs.get("file_names") or []
        if isinstance(file_names_value, str):
            file_names_value = [file_names_value]
        sha256s_value = outputs.get("sha256s") or []
        if isinstance(sha256s_value, str):
            sha256s_value = [sha256s_value]
        artifacts: list[DifyLedgerArtifact] = []
        for index, file_payload in enumerate(file_payloads):
            url = file_payload.get("download_url") or file_payload.get("url") or file_payload.get("source_url")
            raw = self._download(str(url))
            payload_name = str(file_payload.get("name") or "")
            expected_line = line_names[index] if index < len(line_names) else ""
            if not expected_line and payload_name.endswith("经纬度台账.xlsx"):
                expected_line = payload_name[:-len("经纬度台账.xlsx")]
            checked = validate_ledger_bytes(raw, expected_line_name=expected_line or None)
            file_name = (
                str(file_names_value[index]) if index < len(file_names_value) and file_names_value[index]
                else payload_name or f"{expected_line or checked['line_name']}经纬度台账.xlsx"
            )
            artifact_warnings = tuple(str(item) for item in warnings)
            artifacts.append(DifyLedgerArtifact(
                file_name=file_name,
                line_name=str(checked["line_name"]),
                tower_count=int(checked["tower_count"]),
                warnings=artifact_warnings,
                xlsx_bytes=raw,
                sha256=(str(sha256s_value[index]) if index < len(sha256s_value) and sha256s_value[index] else None),
            ))
        return DifyLedgerBatchResult(
            status="succeeded",
            manual_required=False,
            manual_reason="",
            source_file_name=path.name,
            line_full_name=str(outputs.get("line_full_name") or ""),
            voltage_level=str(outputs.get("voltage_level") or ""),
            circuit_type=str(outputs.get("circuit_type") or ""),
            line_names=tuple(item.line_name for item in artifacts),
            warnings=tuple(str(item) for item in warnings),
            artifacts=tuple(artifacts),
            workflow_run_id=data.get("id") or response.get("workflow_run_id"),
            upload_file_id=upload_id,
        )

    def generate_ledger(self, kml_path: str | os.PathLike[str]) -> DifyLedgerResult:
        batch = self.generate_ledgers(kml_path)
        if batch.manual_required:
            raise DifyApiError("MANUAL_INPUT_REQUIRED", batch.manual_reason)
        if len(batch.artifacts) != 1:
            raise DifyApiError("MULTIPLE_OUTPUTS", "该 KML 返回多本台账，请使用 generate_ledgers")
        artifact = batch.artifacts[0]
        return DifyLedgerResult(
            file_name=artifact.file_name,
            line_name=artifact.line_name,
            tower_count=artifact.tower_count,
            warnings=artifact.warnings,
            xlsx_bytes=artifact.xlsx_bytes,
            sha256=artifact.sha256,
            workflow_run_id=batch.workflow_run_id,
        )
