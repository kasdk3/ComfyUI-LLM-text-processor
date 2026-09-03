from __future__ import annotations

import fnmatch
import json
import platform
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import stat
import tarfile
import shutil


LLAMA_CPP_RELEASE_TAG = "b10472"
RELEASE_API_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class LlamaCliPaths:
    cli: Path


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli.exe",
        "ggml-cuda.dll",
        "cudart64_13.dll",
    ),
)

# Linux CPU spec
LINUX_X64 = PlatformSpec(
    key="linux-x64",
    cli_executable="llama-cli",
    asset_patterns=(
        "llama-*-bin-linux-x86_64.tar.gz",
        "llama-*-bin-linux-x86_64.tgz",
        "llama-*-bin-linux-x86_64.zip",
        "llama-*-bin-linux-x64.tar.gz",
        "llama-*-bin-linux-x64.tgz",
        "llama-*-bin-linux-x64.zip",
    ),
    required_files=(
        "llama-cli",
        "libggml.so",
    ),
)

# Linux CUDA spec (heuristic support)
LINUX_X64_CUDA_13 = PlatformSpec(
    key="linux-x64-cuda13",
    cli_executable="llama-cli",
    asset_patterns=(
        "llama-*-bin-linux-cuda-13*-x86_64.tar.gz",
        "llama-*-bin-linux-cuda-13*-x86_64.tgz",
        "llama-*-bin-linux-cuda-13*-x86_64.zip",
        "llama-*-bin-linux-cuda-13*-x64.tar.gz",
        "llama-*-bin-linux-cuda-13*-x64.tgz",
        "llama-*-bin-linux-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli",
        # the CUDA runtime is typically provided by the system, but include common lib names
        "libggml-cuda.so",
    ),
)


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13

    if system == "linux":
        # Allow forcing platform via env var LLC_PLATFORM
        forced = os.environ.get("LLAMA_CPP_PLATFORM") or os.environ.get("LLC_PLATFORM")
        if forced == "linux-x64-cuda13":
            return LINUX_X64_CUDA_13
        if forced == "linux-x64":
            return LINUX_X64

        # Heuristic detection for CUDA: presence of /usr/local/cuda or nvidia-smi on PATH
        if Path("/usr/local/cuda").exists() or shutil.which("nvidia-smi"):
            # Prefer CUDA build if device/runtime present; change if you prefer CPU default
            return LINUX_X64_CUDA_13
        return LINUX_X64

    raise RuntimeError(
        "Automatic llama.cpp binary download currently supports Windows x64 CUDA 13 and Linux x64 (CPU/CUDA). "
        "Other platforms are intentionally isolated behind the platform mapping for future support."
    )


def _json_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_size(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size is not None else None
        downloaded = 0
        chunk_size = 1024 * 256
        started_at = time.monotonic()
        last_reported_at = started_at

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported_at < 1.0:
                    continue

                elapsed = max(now - started_at, 0.001)
                speed = downloaded / elapsed
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({percent:.1f}%) at {_format_size(speed)}/s"
                    )
                else:
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} at {_format_size(speed)}/s"
                    )
                last_reported_at = now

        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed
        if total_size:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} / {_format_size(total_size)} "
                f"(100.0%) at {_format_size(speed)}/s"
            )
        else:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} at {_format_size(speed)}/s"
            )


def _select_assets(release: dict, spec: PlatformSpec) -> list[dict]:
    assets = release.get("assets", [])
    selected = []
    used_names = set()

    # Match explicit release asset names so a future platform can add patterns
    # without changing the download/extract pipeline.
    for pattern in spec.asset_patterns:
        matches = [
            asset for asset in assets
            if fnmatch.fnmatch(asset.get("name", "").lower(), pattern.lower())
        ]
        if not matches:
            raise RuntimeError(f"Could not find llama.cpp release asset matching: {pattern}")
        asset = sorted(matches, key=lambda item: item.get("name", ""))[0]
        if asset["name"] not in used_names:
            selected.append(asset)
            used_names.add(asset["name"])
    return selected


def _find_file(install_dir: Path, name: str) -> Path | None:
    for path in install_dir.rglob(name):
        if path.is_file():
            return path
    return None


def _find_cli_paths(install_dir: Path, spec: PlatformSpec) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    return LlamaCliPaths(cli=cli)


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return _find_cli_paths(install_dir, spec) is not None and _has_required_files(install_dir, spec)


def _existing_install(spec: PlatformSpec) -> LlamaCliPaths | None:
    install_dir = VENDOR_ROOT / LLAMA_CPP_RELEASE_TAG / spec.key
    return _find_cli_paths(install_dir, spec) if _is_complete_install(install_dir, spec) else None


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    # Prevent path traversal (see commons) — ensure members are inside path
    for member in tar.getmembers():
        member_path = Path(path) / member.name
        if not str(member_path.resolve()).startswith(str(path.resolve())):
            raise RuntimeError("Tar archive contains files outside target directory")
    tar.extractall(path)


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="llm-text-processor-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[LLM Text Processor] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)

            name_lower = asset["name"].lower()
            try:
                if name_lower.endswith(".zip"):
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(install_dir)
                elif name_lower.endswith((".tar.gz", ".tgz")):
                    with tarfile.open(archive_path, "r:gz") as tar:
                        _safe_extract_tar(tar, install_dir)
                elif name_lower.endswith(".tar"):
                    with tarfile.open(archive_path, "r:") as tar:
                        _safe_extract_tar(tar, install_dir)
                else:
                    # fallback: try zip then tar
                    try:
                        with zipfile.ZipFile(archive_path) as archive:
                            archive.extractall(install_dir)
                    except zipfile.BadZipFile:
                        with tarfile.open(archive_path, "r:*") as tar:
                            _safe_extract_tar(tar, install_dir)
            except Exception as ex:
                raise RuntimeError(f"Failed to extract {archive_path}: {ex}")

        # Ensure CLI is executable on POSIX systems
        for path in install_dir.rglob("*"):
            if path.is_file() and path.name in { "llama-cli", "llama-cli.exe" }:
                try:
                    mode = path.stat().st_mode
                    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception:
                    # best-effort; ignore permission errors here and allow later checks to fail
                    pass


def ensure_llama_cli_paths() -> LlamaCliPaths:
    spec = _platform_spec()
    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or LLAMA_CPP_RELEASE_TAG
    install_dir = VENDOR_ROOT / tag / spec.key

    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(f"Completed install has incomplete CLI executables: {install_dir}")
        return paths

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    paths = _find_cli_paths(install_dir, spec)
    if paths is None:
        raise RuntimeError(
            f"Downloaded llama.cpp assets but could not find CLI executables in {install_dir}"
        )
    if not _has_required_files(install_dir, spec):
        missing = [
            name for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(f"Downloaded llama.cpp assets are incomplete; missing: {', '.join(missing)}")

    return paths
