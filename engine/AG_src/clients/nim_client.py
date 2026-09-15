"""
nim_client.py
NVIDIA NIM (BioNEMO) REST API 클라이언트 모듈

표준 라이브러리(urllib)만 사용하여 NVIDIA NIM 클라우드 API에 연결.
재시도 로직, 지수 백오프, 헬스체크, dry-run 모드(API 키 없을 때)를 지원.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 기본 상수
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://health.api.nvidia.com/v1"
DEFAULT_TIMEOUT = 60       # 초
DEFAULT_MAX_RETRIES = 3
BACKOFF_BASE = 1.0         # 지수 백오프 기본값 (1s, 2s, 4s)


# ---------------------------------------------------------------------------
# 커스텀 예외 클래스
# ---------------------------------------------------------------------------

class NIMAPIError(Exception):
    """NVIDIA NIM API 일반 오류."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class NIMTimeoutError(NIMAPIError):
    """API 요청 타임아웃 오류."""


class NIMAuthError(NIMAPIError):
    """API 인증 오류 (잘못된 키 또는 키 없음)."""


# ---------------------------------------------------------------------------
# 베이스 클라이언트
# ---------------------------------------------------------------------------

class NIMClient:
    """NVIDIA NIM REST API 베이스 클라이언트.

    API 키가 없으면 dry-run 모드로 동작하여 목(mock) 데이터를 반환한다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        # 환경 변수 우선, 인자로 덮어쓸 수 있음
        self.api_key = api_key or os.environ.get("NVIDIA_NIM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = self.api_key is None  # API 키 없으면 dry-run 모드

        if self.dry_run:
            logger.warning(
                "NVIDIA_NIM_API_KEY 미설정 - dry-run 모드로 목 데이터를 반환합니다."
            )

    # ------------------------------------------------------------------
    # 내부 HTTP 요청 (재시도 + 지수 백오프)
    # ------------------------------------------------------------------

    def _request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        method: str = "POST",
    ) -> dict[str, Any]:
        """HTTP 요청을 보내고 JSON 응답을 반환한다. 실패 시 재시도."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        last_error: Exception = RuntimeError("재시도 횟수 초과")

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method=method
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)

            except urllib.error.HTTPError as exc:
                # 인증 오류는 즉시 실패 처리
                if exc.code in (401, 403):
                    raise NIMAuthError(
                        f"인증 실패 (HTTP {exc.code}): {exc.reason}",
                        status_code=exc.code,
                    ) from exc
                last_error = NIMAPIError(
                    f"HTTP 오류 {exc.code}: {exc.reason}", status_code=exc.code
                )

            except urllib.error.URLError as exc:
                if "timed out" in str(exc.reason).lower():
                    last_error = NIMTimeoutError(f"요청 타임아웃: {url}")
                else:
                    last_error = NIMAPIError(f"연결 오류: {exc.reason}")

            # 지수 백오프 대기 (마지막 시도 후에는 대기 생략)
            if attempt < self.max_retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                logger.debug("재시도 %d/%d - %.1f초 대기", attempt + 1, self.max_retries, wait)
                time.sleep(wait)

        raise last_error

    # ------------------------------------------------------------------
    # 헬스체크
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """엔드포인트 연결 가능 여부를 확인한다."""
        if self.dry_run:
            logger.info("dry-run 모드: health_check=True(가정)")
            return True
        try:
            url = self.base_url
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10):
                return True
        except Exception as exc:
            logger.warning("헬스체크 실패: %s", exc)
            return False

    def __repr__(self) -> str:
        mode = "dry-run" if self.dry_run else "live"
        return f"{self.__class__.__name__}(base_url={self.base_url!r}, mode={mode})"


# ---------------------------------------------------------------------------
# ESMFold 클라이언트 - 빠른 단백질 구조 예측 (QC용)
# ---------------------------------------------------------------------------

class ESMFoldClient(NIMClient):
    """ESMFold: 아미노산 시퀀스 → 3D 구조 예측."""

    ENDPOINT = "/biology/nvidia/esmfold"

    def predict_structure(self, sequence: str) -> dict[str, Any]:
        """단백질 시퀀스로부터 PDB 구조와 pLDDT 점수를 예측한다.

        Args:
            sequence: 단일 문자 아미노산 시퀀스 (예: "MKTLL...")

        Returns:
            {"pdb": str, "plddt": list[float]}
        """
        if self.dry_run:
            return {"pdb": f"ATOM  DRY-RUN-{sequence[:5]}", "plddt": [85.0] * len(sequence)}

        payload = {"sequence": sequence}
        return self._request(self.ENDPOINT, payload)


# ---------------------------------------------------------------------------
# DiffDock 클라이언트 - 분자 도킹
# ---------------------------------------------------------------------------

class DiffDockClient(NIMClient):
    """DiffDock: 단백질-리간드 도킹 포즈 예측."""

    ENDPOINT = "/biology/mit/diffdock"

    def dock(
        self,
        protein_pdb: str,
        ligand_pdb: str,
        n_poses: int = 10,
    ) -> list[dict[str, Any]]:
        """단백질과 리간드의 도킹 포즈를 예측한다.

        Args:
            protein_pdb: 수용체 단백질 PDB 문자열
            ligand_pdb:  리간드 PDB 또는 SMILES 문자열
            n_poses:     반환할 포즈 수

        Returns:
            [{"pose_pdb": str, "score": float}, ...]  점수 내림차순 정렬
        """
        if self.dry_run:
            return [{"pose_pdb": f"MOCK-POSE-{i}", "score": 1.0 / (i + 1)} for i in range(n_poses)]

        payload = {"protein_pdb": protein_pdb, "ligand": ligand_pdb, "num_poses": n_poses}
        return self._request(self.ENDPOINT, payload)


# ---------------------------------------------------------------------------
# RFdiffusion 클라이언트 - 신규 백본 구조 생성
# ---------------------------------------------------------------------------

class RFdiffusionClient(NIMClient):
    """RFdiffusion: 드 노보(de novo) 단백질 백본 생성."""

    ENDPOINT = "/biology/ipd/rfdiffusion/generate"

    def generate_backbone(
        self,
        contigs: str,
        hotspot_res: list[str],
        n_designs: int = 5,
    ) -> list[str]:
        """지정 조건(contigs, hotspot)에 맞는 단백질 백본 PDB를 생성한다.

        Args:
            contigs:     RFdiffusion 형식의 contig 문자열 (예: "A1-50/0 50-100")
            hotspot_res: 핫스팟 잔기 목록 (예: ["A30", "A31"])
            n_designs:   생성할 구조 수

        Returns:
            PDB 문자열 목록
        """
        if self.dry_run:
            return [f"MOCK-BACKBONE-{i}-contigs={contigs}" for i in range(n_designs)]

        payload = {"contigs": contigs, "hotspot_res": hotspot_res, "diffusion_steps": n_designs}
        result = self._request(self.ENDPOINT, payload)
        return result if isinstance(result, list) else result.get("pdbs", [])


# ---------------------------------------------------------------------------
# ProteinMPNN 클라이언트 - 역폴딩(백본 → 시퀀스)
# ---------------------------------------------------------------------------

class ProteinMPNNClient(NIMClient):
    """ProteinMPNN: 백본 구조에 맞는 아미노산 시퀀스 설계."""

    ENDPOINT = "/biology/ipd/proteinmpnn/predict"

    def design_sequences(
        self,
        backbone_pdb: str,
        n_seqs: int = 8,
        temperature: float = 0.1,
    ) -> list[dict[str, Any]]:
        """주어진 백본 구조에 최적화된 시퀀스를 설계한다.

        Args:
            backbone_pdb: 백본 PDB 문자열
            n_seqs:       생성할 시퀀스 수
            temperature:  샘플링 온도 (낮을수록 결정론적)

        Returns:
            [{"sequence": str, "score": float}, ...]  점수 오름차순 정렬
        """
        if self.dry_run:
            return [{"sequence": f"MOCK-SEQ-{i}", "score": -1.0 * i} for i in range(n_seqs)]

        payload = {"pdb": backbone_pdb, "num_seq_per_target": n_seqs, "sampling_temp": temperature}
        return self._request(self.ENDPOINT, payload)


# ---------------------------------------------------------------------------
# NIMClientRegistry - 팩토리 + tool_registry.yaml 연동
# ---------------------------------------------------------------------------

class NIMClientRegistry:
    """tool_registry.yaml 설정을 읽어 NIM 클라이언트를 생성하고 관리하는 팩토리."""

    # 도구 이름 → 클라이언트 클래스 매핑
    _CLIENT_MAP: dict[str, type[NIMClient]] = {
        "esmfold": ESMFoldClient,
        "diffdock": DiffDockClient,
        "rfdiffusion": RFdiffusionClient,
        "proteinmpnn": ProteinMPNNClient,
    }

    def __init__(
        self,
        registry_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("NVIDIA_NIM_API_KEY")
        self._clients: dict[str, NIMClient] = {}

        # tool_registry.yaml 로드 (경로 미지정 시 기본 위치 사용)
        config_path = registry_path or str(
            Path(__file__).parent.parent / "config" / "tool_registry.yaml"
        )
        self._config = self._load_registry(config_path)
        self._init_clients()

    # ------------------------------------------------------------------

    @staticmethod
    def _load_registry(path: str) -> dict[str, Any]:
        """YAML 레지스트리 파일을 로드한다."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("tool_registry.yaml 없음: %s - 기본값 사용", path)
            return {}
        except Exception as exc:
            logger.error("레지스트리 로드 실패: %s", exc)
            return {}

    def _init_clients(self) -> None:
        """레지스트리의 API 도구 목록을 바탕으로 클라이언트를 초기화한다."""
        api_tools: dict[str, Any] = self._config.get("tools", {}).get("api", {})

        for tool_name, client_cls in self._CLIENT_MAP.items():
            tool_cfg = api_tools.get(tool_name, {})
            timeout = tool_cfg.get("timeout_sec", DEFAULT_TIMEOUT)
            self._clients[tool_name] = client_cls(
                api_key=self.api_key,
                timeout=timeout,
            )
            logger.debug("클라이언트 등록: %s (timeout=%ds)", tool_name, timeout)

    # ------------------------------------------------------------------

    def get_client(self, tool_name: str) -> NIMClient:
        """도구 이름으로 클라이언트를 반환한다.

        Args:
            tool_name: "esmfold" | "diffdock" | "rfdiffusion" | "proteinmpnn"

        Raises:
            KeyError: 알 수 없는 도구 이름
        """
        if tool_name not in self._clients:
            available = ", ".join(self._clients.keys())
            raise KeyError(f"알 수 없는 도구: {tool_name!r}. 사용 가능: {available}")
        return self._clients[tool_name]

    def health_check_all(self) -> dict[str, bool]:
        """등록된 모든 클라이언트의 헬스체크 결과를 반환한다."""
        return {name: client.health_check() for name, client in self._clients.items()}
