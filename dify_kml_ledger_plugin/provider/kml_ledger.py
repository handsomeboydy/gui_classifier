from typing import Any

from dify_plugin import ToolProvider


class KmlLedgerProvider(ToolProvider):
    """Provider entrypoint required by Dify before loading the tool class."""

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        # This converter has no third-party credentials.
        return None
