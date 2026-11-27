"""Native Host RPC handlers for FvF Bellows."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageOps

try:
    from .bellows_ethics import GATEKEEPER_MODEL_ID, PROHIBITED_POLICY_LIST, check_bellows_ethics
    from .bellows_trainer import BELLOWS_CACHE_DIR, IMAGE_TIERS, TEXT_TIERS, get_power_status
except Exception:
    from bellows_ethics import GATEKEEPER_MODEL_ID, PROHIBITED_POLICY_LIST, check_bellows_ethics
    from bellows_trainer import BELLOWS_CACHE_DIR, IMAGE_TIERS, TEXT_TIERS, get_power_status

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_ROOT / "native" / "bellows_jobs"
ACTIVE_JOB_PTR = JOBS_DIR / "active_job.json"
HF_BASE_BASE_URL = "https://huggingface.co/datasets/LuckyCow/fvf-bellows-base/resolve/main"

ACTIVE_JOB_STATES = {"initializing", "downloading_backbone", "unpacking_dataset", "running", "exporting"}


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return bool(ok and code.value == 259)  # STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_latest_status(job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    target_dir: Optional[Path] = None
    if job_id:
        cand = (JOBS_DIR / str(job_id)).resolve()
        if JOBS_DIR.resolve() in cand.parents and cand.is_dir():
            target_dir = cand
    elif ACTIVE_JOB_PTR.is_file():
        try:
            ptr = json.loads(ACTIVE_JOB_PTR.read_text(encoding="utf-8"))
            cand = Path(ptr.get("job_dir", ""))
            if cand.is_dir():
                target_dir = cand
        except Exception:
            pass

    if target_dir is None:
        return None
    sp = target_dir / "status.json"
    if not sp.is_file():
        return None
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return None

    st["job_dir"] = str(target_dir)
    if st.get("state") in ACTIVE_JOB_STATES and not _pid_alive(st.get("pid")):
        st["state"] = "failed"
        st["phase"] = "failed"
        st["phase_detail"] = "Training process ended unexpectedly before reporting completion."
    return st


def is_bellows_training_active() -> bool:
    st = _read_latest_status(None)
    return bool(st and st.get("state") in ACTIVE_JOB_STATES and _pid_alive(st.get("pid")))


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return round(total / (1024 * 1024), 1)


def _get_heavy_processes(limit: int = 8) -> List[Dict[str, Any]]:
    procs: Dict[int, Dict[str, Any]] = {}
    try:
        if sys.platform == "win32":
            cmd = (
                "Get-Process | Sort-Object WorkingSet64 -Descending | "
                "Select-Object -First 20 Id, ProcessName, @{Name='RamMB';Expression={[math]::Round($_.WorkingSet64/1MB,1)}} | "
                "ConvertTo-Json -Compress"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                timeout=3.5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            rows = json.loads(out) if out.strip() else []
            if isinstance(rows, dict):
                rows = [rows]
            # Merge duplicate app names cleanly (e.g. Chrome/Antigravity)
            by_name: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                name = str(r.get("ProcessName") or "process")
                ram = float(r.get("RamMB") or 0.0)
                if name.lower() in ("system", "idle", "memory compression", "svchost", "dwm", "python"):
                    continue
                if ram < 150.0:
                    continue
                cur = by_name.setdefault(name, {"name": name, "ram_mb": 0.0, "vram_mb": 0.0, "pids": 0})
                cur["ram_mb"] = round(cur["ram_mb"] + ram, 1)
                cur["pids"] += 1
            for item in by_name.values():
                procs[hash(item["name"])] = item
        elif Path("/proc").is_dir():
            ps_out = subprocess.check_output(["ps", "-eo", "pid,rss,comm", "--sort=-rss"], text=True, timeout=2.5)
            by_name: Dict[str, Dict[str, Any]] = {}
            for ln in ps_out.splitlines()[1:25]:
                parts = ln.split(None, 2)
                if len(parts) < 3:
                    continue
                ram = round(int(parts[1]) / 1024.0, 1)
                if ram < 150.0:
                    continue
                name = parts[2].strip()
                cur = by_name.setdefault(name, {"name": name, "ram_mb": 0.0, "vram_mb": 0.0, "pids": 0})
                cur["ram_mb"] = round(cur["ram_mb"] + ram, 1)
                cur["pids"] += 1
            for item in by_name.values():
                procs[hash(item["name"])] = item
    except Exception:
        pass

    # Query NVIDIA VRAM consumers when nvidia-smi exists
    try:
        nout = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
        vram_map: Dict[str, float] = {}
        for ln in nout.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) == 2 and parts[1].isdigit():
                pname = Path(parts[0]).stem
                vram_map[pname] = vram_map.get(pname, 0.0) + float(parts[1])
        for pname, vmb in vram_map.items():
            matched = next((x for x in procs.values() if x["name"].lower() == pname.lower()), None)
            if matched:
                matched["vram_mb"] = round(vmb, 1)
            elif vmb >= 100:
                procs[hash("vram_" + pname)] = {"name": pname, "ram_mb": 0.0, "vram_mb": round(vmb, 1), "pids": 1}
    except Exception:
        pass

    ranked = sorted(procs.values(), key=lambda x: (x["vram_mb"] * 2.0 + x["ram_mb"]), reverse=True)
    return ranked[:limit]


def _probe_training_hardware(sys_mem: Dict[str, float]) -> Dict[str, Any]:
    has_gpu = False
    hw_name = "CPU"
    backend = "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            has_gpu = True
            hw_name = torch.cuda.get_device_name(0)
            backend = "CUDA"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_gpu = True
            hw_name = "Apple Silicon (MPS)"
            backend = "MPS"
        else:
            try:
                import torch_directml

                has_gpu = True
                hw_name = torch_directml.device_name(0)
                backend = "DirectML"
            except Exception:
                pass
    except Exception:
        pass

    if not has_gpu and float(sys_mem.get("vram_total_mb", 0.0)) > 0:
        has_gpu = True
        hw_name = "GPU"
        backend = "GPU"

    return {
        "has_gpu": has_gpu,
        "name": hw_name,
        "backend": backend,
        "cpu_threads": os.cpu_count() or 4,
    }


def _fmt_duration(minutes: int) -> str:
    if minutes < 60:
        return f"~{minutes} min"
    hrs = minutes / 60.0
    if hrs < 1.5:
        return "~1–1.5 hours"
    return f"~{hrs:.1f} hours"


def bellows_preflight(sys_mem: Dict[str, float], request_id: Optional[str] = None) -> Dict[str, Any]:
    power = get_power_status()
    hw = _probe_training_hardware(sys_mem)
    disk = shutil.disk_usage(str(REPO_ROOT))
    free_gb = round(disk.free / (1024 ** 3), 2)

    img_tiers_out = {}
    for k, v in IMAGE_TIERS.items():
        est_min = v["est_gpu_min"] if hw["has_gpu"] else v["est_cpu_min"]
        img_tiers_out[k] = {**v, "est_minutes": est_min, "est_text": _fmt_duration(est_min)}

    txt_tiers_out = {}
    for k, v in TEXT_TIERS.items():
        est_min = v["est_gpu_min"] if hw["has_gpu"] else v["est_cpu_min"]
        txt_tiers_out[k] = {**v, "est_minutes": est_min, "est_text": _fmt_duration(est_min)}

    active_st = _read_latest_status(None)
    resp: Dict[str, Any] = {
        "ok": True,
        "type": "bellowsPreflight",
        "power": power,
        "memory": sys_mem,
        "free_disk_gb": free_gb,
        "hardware": hw,
        "heavy_processes": _get_heavy_processes(limit=8),
        "disk_requirements": {
            "image_gb": 1.2,
            "text_gb": 0.9,
            "image_label": "~1.2 GB temporary free space (base images + gatekeeper + backbone cache)",
            "text_label": "~0.9 GB temporary free space (base text + gatekeeper + encoder cache)",
        },
        "cache_size_mb": _dir_size_mb(BELLOWS_CACHE_DIR),
        "tiers": {"image": img_tiers_out, "text": txt_tiers_out},
        "prohibited_policies": PROHIBITED_POLICY_LIST,
        "active_job": active_st,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_check_ethics(
    msg: Dict[str, Any],
    send_fn: Callable[[Dict[str, Any]], None],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    modality = str(msg.get("modality") or "image")
    prompt = str(msg.get("prompt") or "")

    def on_prog(ev: Dict[str, Any]) -> None:
        send_fn({
            "ok": True,
            "type": "bellowsProgress",
            "stage": "ethics",
            **ev,
            "timestamp": int(time.time() * 1000),
        })

    result = check_bellows_ethics(modality, prompt, BELLOWS_CACHE_DIR, progress_cb=on_prog)
    resp: Dict[str, Any] = {
        "ok": True,
        "type": "bellowsEthicsResult",
        "gatekeeper_model": GATEKEEPER_MODEL_ID,
        **result,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def _download_with_progress(url: str, dest: Path, stage_name: str, send_fn: Callable[[Dict[str, Any]], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "FireVsFire-Bellows/1.0"})
    last_emit = 0.0
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_emit >= 0.25 or (total > 0 and done >= total):
                    last_emit = now
                    pct = round(100.0 * done / total, 1) if total > 0 else 50.0
                    send_fn({
                        "ok": True,
                        "type": "bellowsProgress",
                        "stage": stage_name,
                        "bytesDownloaded": done,
                        "totalBytes": total,
                        "percent": pct,
                        "message": f"Downloading base web dataset ({done / (1024*1024):.1f} / {max(done, total) / (1024*1024):.1f} MB)...",
                    })
    part.replace(dest)


def _count_job_samples(job_dir: Path, modality: str) -> Dict[str, int]:
    ds = job_dir / "dataset"
    if modality == "image":
        pos = len([p for p in (ds / "positives").glob("*") if p.is_file()]) if (ds / "positives").is_dir() else 0
        hn = len([p for p in (ds / "hard_negatives").glob("*") if p.is_file()]) if (ds / "hard_negatives").is_dir() else 0
        base = 2400 if (BELLOWS_CACHE_DIR / "base_datasets" / "base_images_v1.zip").is_file() else 0
        return {"positives": pos, "hard_negatives": hn, "base_negatives": base}
    pos_f = ds / "text_positives.jsonl"
    hn_f = ds / "text_hard_negatives.jsonl"
    pos = len([ln for ln in pos_f.read_text(encoding="utf-8").splitlines() if ln.strip()]) if pos_f.is_file() else 0
    hn = len([ln for ln in hn_f.read_text(encoding="utf-8").splitlines() if ln.strip()]) if hn_f.is_file() else 0
    base = 5600 if (BELLOWS_CACHE_DIR / "base_datasets" / "base_text_v1.parquet").is_file() else 0
    return {"positives": pos, "hard_negatives": hn, "base_negatives": base}


def bellows_prepare_session(
    msg: Dict[str, Any],
    send_fn: Callable[[Dict[str, Any]], None],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    modality = "text" if str(msg.get("modality") or "image").lower() == "text" else "image"
    prompt = str(msg.get("prompt") or "custom")
    job_id = str(msg.get("jobId") or f"bellows_{int(time.time())}")
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", job_id) or f"bellows_{int(time.time())}"

    job_dir = JOBS_DIR / job_id
    (job_dir / "dataset" / "positives").mkdir(parents=True, exist_ok=True)
    (job_dir / "dataset" / "hard_negatives").mkdir(parents=True, exist_ok=True)

    base_filename = "base_images_v1.zip" if modality == "image" else "base_text_v1.parquet"
    base_dest = BELLOWS_CACHE_DIR / "base_datasets" / base_filename
    base_url = f"{HF_BASE_BASE_URL}/{base_filename}"

    if not base_dest.is_file() or base_dest.stat().st_size == 0:
        _download_with_progress(base_url, base_dest, "base_dataset", send_fn)

    session_meta = {"job_id": job_id, "modality": modality, "prompt": prompt}
    (job_dir / "session.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")

    resp: Dict[str, Any] = {
        "ok": True,
        "type": "bellowsSessionReady",
        "jobId": job_id,
        "modality": modality,
        "baseDatasetPath": str(base_dest),
        "counts": _count_job_samples(job_dir, modality),
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_add_samples(msg: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(msg.get("jobId") or ""))
    if not job_id or not (JOBS_DIR / job_id).is_dir():
        return {"ok": False, "error": "Invalid Bellows session jobId", **({"requestId": request_id} if request_id else {})}

    job_dir = JOBS_DIR / job_id
    modality = "text" if str(msg.get("modality") or "image").lower() == "text" else "image"
    kind = "hard_negatives" if str(msg.get("kind") or "positive").startswith("hard") else "positives"
    items = msg.get("samples") or []
    added_ids: List[str] = []

    if modality == "image":
        target_dir = job_dir / "dataset" / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            raw_b64 = (item.get("dataUrl") or item.get("data") or "") if isinstance(item, dict) else (item if isinstance(item, str) else "")
            if not raw_b64 or not isinstance(raw_b64, str):
                continue
            try:
                b64_part = raw_b64.split(",", 1)[1] if "," in raw_b64 and raw_b64.startswith("data:") else raw_b64
                raw_bytes = base64.b64decode(b64_part)
                with Image.open(BytesIO(raw_bytes)) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    if min(im.size) < 32:
                        continue
                    im_res = im.resize((256, 256), Image.Resampling.BILINEAR)
                    buf = BytesIO()
                    im_res.save(buf, format="JPEG", quality=88)
                    out_bytes = buf.getvalue()
                hsh = hashlib.sha1(out_bytes).hexdigest()[:16]
                out_file = target_dir / f"{hsh}.jpg"
                out_file.write_bytes(out_bytes)
                # Ensure mutual exclusivity if switched between positive and hard negative
                other = "hard_negatives" if kind == "positives" else "positives"
                (job_dir / "dataset" / other / f"{hsh}.jpg").unlink(missing_ok=True)
                added_ids.append(hsh)
            except Exception:
                continue
    else:
        fname = "text_hard_negatives.jsonl" if kind == "hard_negatives" else "text_positives.jsonl"
        other_fname = "text_positives.jsonl" if kind == "hard_negatives" else "text_hard_negatives.jsonl"
        target_f = job_dir / "dataset" / fname
        other_f = job_dir / "dataset" / other_fname
        existing: Dict[str, str] = {}
        if target_f.is_file():
            for ln in target_f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    existing[rec["id"]] = json.dumps(rec, ensure_ascii=False)
        other_existing: Dict[str, str] = {}
        if other_f.is_file():
            for ln in other_f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    other_existing[rec["id"]] = json.dumps(rec, ensure_ascii=False)

        for item in items:
            txt = item.get("text") if isinstance(item, dict) else str(item)
            txt = re.sub(r"\s+", " ", txt or "").strip()
            if len(txt) < 40:
                continue
            hsh = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]
            existing[hsh] = json.dumps({"id": hsh, "text": txt}, ensure_ascii=False)
            other_existing.pop(hsh, None)
            added_ids.append(hsh)

        target_f.write_text("\n".join(existing.values()) + ("\n" if existing else ""), encoding="utf-8")
        other_f.write_text("\n".join(other_existing.values()) + ("\n" if other_existing else ""), encoding="utf-8")

    counts = _count_job_samples(job_dir, modality)
    resp = {
        "ok": True,
        "type": "bellowsSamplesUpdated",
        "jobId": job_id,
        "addedIds": added_ids,
        "counts": counts,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_remove_sample(msg: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(msg.get("jobId") or ""))
    sample_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(msg.get("sampleId") or ""))
    modality = "text" if str(msg.get("modality") or "image").lower() == "text" else "image"
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir() or not sample_id:
        return {"ok": False, "error": "Invalid jobId or sampleId", **({"requestId": request_id} if request_id else {})}

    if modality == "image":
        for folder in ("positives", "hard_negatives"):
            (job_dir / "dataset" / folder / f"{sample_id}.jpg").unlink(missing_ok=True)
    else:
        for fn in ("text_positives.jsonl", "text_hard_negatives.jsonl"):
            fp = job_dir / "dataset" / fn
            if fp.is_file():
                kept = [ln for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln).get("id") != sample_id]
                fp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    resp = {
        "ok": True,
        "type": "bellowsSamplesUpdated",
        "jobId": job_id,
        "counts": _count_job_samples(job_dir, modality),
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_start_training(
    msg: Dict[str, Any],
    on_before_train: Callable[[], None],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    power = get_power_status()
    if not power.get("is_plugged_in", False):
        return {
            "ok": False,
            "error": "Computer must be plugged into wall power (AC) before Bellows training can start.",
            "power": power,
            **({"requestId": request_id} if request_id else {}),
        }

    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(msg.get("jobId") or ""))
    if not job_id or not (JOBS_DIR / job_id).is_dir():
        return {"ok": False, "error": "Bellows dataset session not found.", **({"requestId": request_id} if request_id else {})}

    job_dir = JOBS_DIR / job_id
    modality = "text" if str(msg.get("modality") or "image").lower() == "text" else "image"
    counts = _count_job_samples(job_dir, modality)
    if counts["positives"] < 15 and not bool(msg.get("_allowSmallSmoke", False)):
        return {
            "ok": False,
            "error": f"Please provide at least 15 target examples before training (currently {counts['positives']}).",
            **({"requestId": request_id} if request_id else {}),
        }

    model_name = str(msg.get("modelName") or "My Bellows Filter").strip()[:60]
    label = str(msg.get("label") or msg.get("prompt") or "Custom filter").strip()[:40]
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[:36] or "custom"
    # Avoid colliding with an older model unless intended
    if (REPO_ROOT / "native" / "classifiers" / f"bellows_{slug}_classifier.py").exists():
        slug = f"{slug}_{int(time.time()) % 10000}"

    tier = str(msg.get("tier") or "balanced").lower()
    if tier not in ("swift", "balanced", "thorough"):
        tier = "balanced"
    thresh = float(msg.get("suggestedThreshold", 0.75))
    thresh = max(0.05, min(0.95, thresh))

    cfg = {
        "job_id": job_id,
        "modality": modality,
        "prompt": str(msg.get("prompt") or label),
        "model_name": model_name,
        "label": label,
        "slug": slug,
        "tier": tier,
        "suggested_threshold": thresh,
        "auto_threshold": bool(msg.get("autoThreshold", False)),
        **({"epochs_override": int(msg["epochsOverride"])} if msg.get("epochsOverride") else {}),
    }
    (job_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (job_dir / "stop.flag").unlink(missing_ok=True)

    # Pause normal FvF classification & release warm models before launching trainer
    try:
        on_before_train()
    except Exception:
        pass

    trainer_script = REPO_ROOT / "native" / "bellows_trainer.py"
    log_file = open(job_dir / "train.log", "w", encoding="utf-8")
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "-u", str(trainer_script), "--job-dir", str(job_dir)],
        **popen_kwargs,
    )
    log_file.close()

    init_status = {
        "job_id": job_id,
        "pid": proc.pid,
        "state": "initializing",
        "phase": "initializing",
        "phase_detail": "Starting Bellows training worker...",
        "modality": modality,
        "model_name": model_name,
        "label": label,
        "tier": tier,
        "progress_pct": 0.0,
        "epoch": 0,
        "total_epochs": int(cfg.get("epochs_override") or (IMAGE_TIERS if modality == "image" else TEXT_TIERS)[tier]["epochs"]),
        "step": 0,
        "total_steps": 1,
        "history": [],
        "power_status": get_power_status(),
        "updated_at": time.time(),
    }
    (job_dir / "status.json").write_text(json.dumps(init_status, indent=2), encoding="utf-8")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_JOB_PTR.write_text(json.dumps({"job_id": job_id, "job_dir": str(job_dir), "pid": proc.pid}), encoding="utf-8")

    resp = {
        "ok": True,
        "type": "bellowsTrainingStarted",
        "jobId": job_id,
        "pid": proc.pid,
        "classifierId": f"bellows_{slug}",
        "config": cfg,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_get_status(msg: Dict[str, Any], on_finished_cb: Callable[[], None], request_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = msg.get("jobId")
    st = _read_latest_status(str(job_id) if job_id else None)
    if st and st.get("state") in ("completed", "stopped_low_battery", "stopped_early"):
        try:
            on_finished_cb()
        except Exception:
            pass

    resp: Dict[str, Any] = {
        "ok": True,
        "type": "bellowsStatus",
        "active": bool(st and st.get("state") in ACTIVE_JOB_STATES),
        "status": st,
        "power": get_power_status(),
        "cache_size_mb": _dir_size_mb(BELLOWS_CACHE_DIR),
        "jobs_size_mb": _dir_size_mb(JOBS_DIR),
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def bellows_stop_training(msg: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    st = _read_latest_status(msg.get("jobId"))
    if not st or not st.get("job_dir"):
        return {"ok": False, "error": "No active training job found.", **({"requestId": request_id} if request_id else {})}
    flag = Path(st["job_dir"]) / "stop.flag"
    flag.write_text("stop\n", encoding="utf-8")
    resp = {"ok": True, "type": "bellowsStopRequested", "jobId": st.get("job_id"), **({"requestId": request_id} if request_id else {})}
    return resp


def bellows_cleanup_storage(msg: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    delete_cache = bool(msg.get("deleteCache", True))
    delete_datasets = bool(msg.get("deleteDatasets", True))
    freed_mb = 0.0
    if delete_cache and BELLOWS_CACHE_DIR.exists():
        freed_mb += _dir_size_mb(BELLOWS_CACHE_DIR)
        shutil.rmtree(BELLOWS_CACHE_DIR, ignore_errors=True)
    if delete_datasets and JOBS_DIR.exists():
        for jd in JOBS_DIR.iterdir():
            if jd.is_dir():
                ds = jd / "dataset"
                if ds.exists():
                    freed_mb += _dir_size_mb(ds)
                    shutil.rmtree(ds, ignore_errors=True)
    resp = {
        "ok": True,
        "type": "bellowsCleanupComplete",
        "freed_mb": round(freed_mb, 1),
        "cache_size_mb": _dir_size_mb(BELLOWS_CACHE_DIR),
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp
