"""
Base Tool for NVIDIA NIM API Wrappers
======================================
Defines the agentic tool interface that all API tool wrappers inherit from.
Handles API key loading, HTTP retries, and a uniform ToolResult return type.

Design note: This module deliberately does NOT import from PRST_N_FM/bionemo/.
It re-implements the same patterns independently so the agentic layer is
self-contained.

Retry policy: 429 / 500 / 502 / 503 / 504 -> exponential backoff, max 3 retries.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 2.0  # seconds: 2, 4, 8 ...

_KEY_ENV_VARS: tuple[str, ...] = ("NGC_CLI_API_KEY", "NVIDIA_API_KEY")
_KEY_FILE_NAMES: tuple[str, ...] = ("ngc.key", "molmim.key")


# ---------------------------------------------------------------------------
# ToolResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Uniform return type for all agentic tool executions.

    Attributes:
        success:    True when the API call completed without error.
        data:       Parsed response payload (empty dict on failure).
        error:      Human-readable error message, or None on success.
        elapsed_ms: Wall-clock time of the API round-trip in milliseconds.
        metadata:   Optional extra fields (tool name, endpoint, attempt count …).
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def ok(
        cls,
        data: dict[str, Any],
        elapsed_ms: int = 0,
        **meta: Any,
    ) -> "ToolResult":
        """Return a successful ToolResult."""
        return cls(success=True, data=data, elapsed_ms=elapsed_ms, metadata=meta)

    @classmethod
    def fail(
        cls,
        error: str,
        elapsed_ms: int = 0,
        **meta: Any,
    ) -> "ToolResult":
        """Return a failed ToolResult."""
        return cls(
            success=False,
            data={},
            error=error,
            elapsed_ms=elapsed_ms,
            metadata=meta,
        )

    def raise_on_error(self) -> "ToolResult":
        """Raise RuntimeError if this result represents a failure."""
        if not self.success:
            raise RuntimeError(self.error or "Tool execution failed (unknown reason)")
        return self


# ---------------------------------------------------------------------------
# BaseTool abstract class
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """Abstract base class for all NVIDIA NIM API tool wrappers.

    Subclasses must define:
        name        -- unique tool identifier string
        description -- one-line description used by the agent
        endpoint    -- full HTTPS base URL for the NIM API
        timeout     -- default request timeout in seconds

    And must implement:
        execute(**kwargs) -> ToolResult
    """

    # ---- Override in subclass -------------------------------------------------
    name: str = ""
    description: str = ""
    endpoint: str = ""
    timeout: int = 120

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        """Initialise the tool and resolve the NVIDIA API key.

        Args:
            api_key:     Explicit API key. When None the key is auto-discovered
                         from environment variables, .env files, and key files.
            max_retries: Maximum number of retry attempts for retryable errors.
        """
        self.api_key: str = api_key or self._load_api_key()
        self.max_retries: int = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        All concrete tool methods should delegate here or be called directly.
        The agent dispatcher calls this entry point.
        """

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _load_api_key() -> str:
        """Locate the NVIDIA API key.

        Search order:
          1. Environment variables: NGC_CLI_API_KEY, NVIDIA_API_KEY
          2. .env file next to this module (looks for any *API_KEY=* line)
          3. Key files (ngc.key, molmim.key) in the project root or cwd
        """
        # 1. Environment variables
        for var in _KEY_ENV_VARS:
            val = os.getenv(var, "").strip()
            if val:
                logger.debug("API key loaded from environment variable %s", var)
                return val

        # 2. .env file adjacent to this source file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if not env_file.exists():
            env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "API_KEY=" in line and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and "your-" not in val.lower():
                        logger.debug("API key loaded from .env file %s", env_file)
                        return val

        # 3. Key files in common locations
        search_dirs = [
            Path(__file__).parent.parent.parent,  # project root
            Path.cwd(),
        ]
        for directory in search_dirs:
            for filename in _KEY_FILE_NAMES:
                key_file = directory / filename
                if key_file.exists():
                    val = key_file.read_text(encoding="utf-8").strip()
                    if val:
                        logger.debug("API key loaded from key file %s", key_file)
                        return val

        raise ValueError(
            "NVIDIA API key not found.\n"
            "Option 1: set environment variable NGC_CLI_API_KEY\n"
            "Option 2: add NGC_CLI_API_KEY=<key> to a .env file in the project root\n"
            "Option 3: create a file named ngc.key in the project root"
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Send a POST request and return the parsed JSON response.

        Args:
            path:    URL path appended to self.endpoint (use "" to POST to root).
            payload: JSON-serialisable request body.
            timeout: Per-request timeout override in seconds.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            RuntimeError: On non-retryable HTTP errors or exhausted retries.
        """
        url = f"{self.endpoint.rstrip('/')}/{path}".rstrip("/") if path else self.endpoint
        resp = self._request_with_retry("POST", url, timeout or self.timeout, json=payload)
        return resp.json()

    def _post_timed(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Like _post but also returns elapsed wall-clock time in milliseconds."""
        t0 = time.monotonic()
        data = self._post(path, payload, timeout)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return data, elapsed_ms

    def _request_with_retry(
        self,
        method: str,
        url: str,
        timeout: int,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute an HTTP request with exponential-backoff retries.

        Retries on: 429, 500, 502, 503, 504.
        Respects the Retry-After header for 429 responses.
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(method, url, timeout=timeout, **kwargs)

                if resp.status_code == 200:
                    return resp

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"API error [{resp.status_code}] from {url}: {resp.text[:500]}"
                    )

                # Retryable — compute backoff wait
                wait = _BACKOFF_BASE ** attempt
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "")
                    try:
                        wait = max(wait, float(retry_after))
                    except (ValueError, TypeError):
                        pass

                logger.warning(
                    "HTTP %d from %s — retry %d/%d in %.1fs",
                    resp.status_code,
                    url,
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
                last_exc = RuntimeError(
                    f"API error [{resp.status_code}]: {resp.text[:500]}"
                )

            except requests.exceptions.Timeout as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Timeout on %s — retry %d/%d in %.1fs",
                    url,
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc

            except requests.exceptions.ConnectionError as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Connection error on %s — retry %d/%d in %.1fs: %s",
                    url,
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
                last_exc = exc

        raise RuntimeError(
            f"Request to {url} failed after {self.max_retries} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, endpoint={self.endpoint!r})"
