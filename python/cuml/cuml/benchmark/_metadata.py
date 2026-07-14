#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Metadata collection helpers for benchmark JSON output."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _git_output(args, cwd):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _read_first_cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or None
    return platform.processor() or None


def _read_total_memory_bytes():
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _gpu_hardware_from_nvml():
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if isinstance(name, bytes):
                name = name.decode()
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "total_memory_bytes": int(memory.total),
                    "compute_capability": f"{major}.{minor}",
                }
            )
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()
        return {
            "count": device_count,
            "devices": devices,
            "driver_version": driver_version,
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_hardware_from_nvidia_smi():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None

    devices = []
    driver_version = None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index, name, uuid, memory_mib, driver_version = parts[:5]
        try:
            memory_bytes = int(float(memory_mib) * 1024 * 1024)
            index = int(index)
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "total_memory_bytes": memory_bytes,
                "compute_capability": None,
            }
        )
    return {
        "count": len(devices),
        "devices": devices,
        "driver_version": driver_version,
    }


def _collect_detected_hardware():
    gpu = _gpu_hardware_from_nvml() or _gpu_hardware_from_nvidia_smi()
    if gpu is None:
        gpu = {"count": 0, "devices": [], "driver_version": None}
    return {
        "gpu": {"detected": gpu},
        "cpu": {
            "detected": {
                "model": _read_first_cpu_model(),
                "logical_cores": os.cpu_count(),
                "physical_cores": None,
                "architecture": platform.machine(),
            }
        },
        "memory": {
            "detected": {
                "total_memory_bytes": _read_total_memory_bytes(),
            }
        },
        "os": {
            "detected": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "kernel": platform.version(),
            }
        },
    }


def _run_json_command(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip() or completed.stdout.strip()
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _conda_package_snapshot(packages):
    return [
        {
            "name": package.get("name"),
            "version": package.get("version"),
            "build": package.get("build_string"),
            "channel": package.get("channel"),
        }
        for package in packages
    ]


def _pip_package_snapshot(packages):
    return [
        {
            "name": package.get("name"),
            "version": package.get("version"),
        }
        for package in packages
    ]


def collect_package_snapshot():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        packages, error = _run_json_command(["conda", "list", "--json"])
        if packages is not None:
            return {
                "package_snapshot_source": "conda",
                "conda_prefix": conda_prefix,
                "packages": _conda_package_snapshot(packages),
            }
        conda_error = error
    else:
        conda_error = None

    packages, error = _run_json_command(
        [sys.executable, "-m", "pip", "list", "--format=json"]
    )
    if packages is not None:
        return {
            "package_snapshot_source": "pip",
            "conda_prefix": conda_prefix,
            "packages": _pip_package_snapshot(packages),
            "conda_error": conda_error,
        }
    return {
        "package_snapshot_source": None,
        "conda_prefix": conda_prefix,
        "packages": [],
        "conda_error": conda_error,
        "pip_error": error,
    }


def _parse_override_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_metadata_overrides(overrides):
    metadata = {}
    for override in overrides or []:
        key, separator, value = override.partition("=")
        if not separator or not key:
            raise ValueError(
                "--metadata-override values must use dotted KEY=VALUE syntax"
            )

        current = metadata
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(
                "--metadata-override keys must not contain empty path parts"
            )
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError(
                    f"--metadata-override key '{key}' conflicts with another override"
                )
        current[parts[-1]] = _parse_override_value(value)
    return metadata


def _deep_merge(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def collect_run_metadata(args, status_string, gpu_available, cuml_version):
    git_cwd = os.getcwd()
    git_sha = _git_output(["rev-parse", "HEAD"], cwd=git_cwd)
    git_status = _git_output(["status", "--porcelain"], cwd=git_cwd)
    git_repo = _git_output(["rev-parse", "--show-toplevel"], cwd=git_cwd)

    metadata = {
        "result_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(getattr(args, "_argv", sys.argv[1:])),
            "cwd": git_cwd,
        },
        "host": {
            "hostname": platform.node(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "cuml": {
            "version": cuml_version,
            "git_repo": git_repo,
            "git_sha": git_sha,
            "git_dirty": bool(git_status) if git_status is not None else None,
        },
        "runtime": {
            "status": status_string,
            "gpu_available": gpu_available,
        },
        "environment": collect_package_snapshot(),
        "hardware": _collect_detected_hardware(),
        "config": {
            "path": getattr(args, "config", None),
            "profile": getattr(args, "profile", None),
            "backends": getattr(args, "backends", None),
        },
    }
    return _deep_merge(
        metadata,
        parse_metadata_overrides(getattr(args, "metadata_override", None)),
    )
