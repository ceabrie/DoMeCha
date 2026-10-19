#!/usr/bin/env python3
"""Probe WireGuard and proxy paths, score them, expose the best path."""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROBE_SECONDS = 5
# Throughput dominates; latency/loss nudge the ranking lightly.
W_THROUGHPUT = 0.75
W_RTT = 0.15
W_LOSS = 0.10

_DIR = Path(__file__).resolve().parent
_PATHS_JSON = _DIR / "paths.json"


@dataclass
class PathCandidate:
    id: str
    name: str
    kind: str  # "wireguard" | "proxy"
    config: dict[str, Any]
    bytes_downloaded: int = 0
    throughput_bps: float = 0.0
    rtt_ms: float = math.inf
    loss: float = 1.0  # 0.0 = none, 1.0 = total failure
    score: float = 0.0
    ok: bool = False
    error: str = ""


@dataclass
class BestPath:
    """Selected path for the rest of the project to consume."""

    id: str
    name: str
    kind: str
    config: dict[str, Any]
    score: float
    throughput_bps: float
    rtt_ms: float
    loss: float

    @property
    def curl_args(self) -> list[str]:
        """Extra curl flags to route a download through this path."""
        return _curl_route_args(self.kind, self.config)


class PathProber:
    """
    Loads servers from paths.json, probes them in parallel with curl for
    PROBE_SECONDS, scores by throughput (+ lighter RTT/loss penalties),
    and keeps the winner as best_path.
    """

    def __init__(
        self,
        paths_file: Path | str = _PATHS_JSON,
        probe_url: Optional[str] = None,
        probe_seconds: float = PROBE_SECONDS,
    ) -> None:
        self._paths_file = Path(paths_file)
        self._probe_seconds = probe_seconds
        self._data = self._load()
        self._probe_url = probe_url or self._data.get(
            "probe_url",
            "https://speed.cloudflare.com/__down?bytes=104857600",
        )
        self._candidates: list[PathCandidate] = []
        self._best: Optional[BestPath] = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        with self._paths_file.open(encoding="utf-8") as f:
            return json.load(f)

    def reload(self) -> None:
        self._data = self._load()
        if "probe_url" in self._data and self._probe_url is None:
            self._probe_url = self._data["probe_url"]

    @property
    def best_path(self) -> Optional[BestPath]:
        """Highest-scoring path from the last probe run (or None)."""
        with self._lock:
            return self._best

    @property
    def candidates(self) -> list[PathCandidate]:
        with self._lock:
            return list(self._candidates)

    def probe(self, download_url: Optional[str] = None) -> Optional[BestPath]:
        """
        Probe every WireGuard and proxy entry in parallel for probe_seconds.
        download_url is the file the user will download (frontend can set later).
        """
        url = download_url or self._probe_url
        servers = self._collect_servers()
        if not servers:
            with self._lock:
                self._candidates = []
                self._best = None
            return None

        results: list[PathCandidate] = []
        with ThreadPoolExecutor(max_workers=max(1, len(servers))) as pool:
            futures = {
                pool.submit(self._probe_one, cand, url): cand for cand in servers
            }
            for fut in as_completed(futures):
                results.append(fut.result())

        self._score_all(results)
        results.sort(key=lambda c: c.score, reverse=True)
        winner = next((c for c in results if c.ok), None)

        best: Optional[BestPath] = None
        if winner is not None:
            best = BestPath(
                id=winner.id,
                name=winner.name,
                kind=winner.kind,
                config=dict(winner.config),
                score=winner.score,
                throughput_bps=winner.throughput_bps,
                rtt_ms=winner.rtt_ms,
                loss=winner.loss,
            )

        with self._lock:
            self._candidates = results
            self._best = best
        return best

    def _collect_servers(self) -> list[PathCandidate]:
        out: list[PathCandidate] = []
        for entry in self._data.get("wireguard", []):
            out.append(
                PathCandidate(
                    id=entry["id"],
                    name=entry.get("name", entry["id"]),
                    kind="wireguard",
                    config=dict(entry),
                )
            )
        for entry in self._data.get("proxy", []):
            out.append(
                PathCandidate(
                    id=entry["id"],
                    name=entry.get("name", entry["id"]),
                    kind="proxy",
                    config=dict(entry),
                )
            )
        return out

    def _probe_one(self, cand: PathCandidate, url: str) -> PathCandidate:
        cmd = [
            "curl",
            "--silent",
            "--show-error",
            "--location",
            "--output",
            "/dev/null",
            "--max-time",
            str(self._probe_seconds),
            "--connect-timeout",
            "3",
            "--write-out",
            "%{size_download} %{speed_download} %{time_connect} %{time_starttransfer} %{http_code}",
            *(_curl_route_args(cand.kind, cand.config)),
            url,
        ]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._probe_seconds + 4,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            cand.error = str(exc)
            cand.loss = 1.0
            return cand

        elapsed = max(time.monotonic() - t0, 1e-6)
        parts = proc.stdout.strip().split()
        if len(parts) < 5:
            cand.error = (proc.stderr or "curl produced no metrics").strip()
            cand.loss = 1.0
            return cand

        try:
            size = float(parts[0])
            speed = float(parts[1])  # bytes/sec (curl average)
            t_connect = float(parts[2])
            t_start = float(parts[3])
            http_code = int(float(parts[4]))
        except ValueError:
            cand.error = f"bad curl metrics: {proc.stdout!r}"
            cand.loss = 1.0
            return cand

        # Prefer curl's reported speed; fall back to size / wall time.
        throughput = speed if speed > 0 else size / elapsed
        rtt = t_connect if t_connect > 0 else t_start

        cand.bytes_downloaded = int(size)
        cand.throughput_bps = max(0.0, throughput)
        cand.rtt_ms = rtt * 1000.0 if rtt > 0 else math.inf
        # Partial download within the time window is expected; treat as success
        # if we got any payload. HTTP errors / zero bytes count as loss.
        if size > 0 and (200 <= http_code < 400 or http_code == 0):
            cand.ok = True
            cand.loss = 0.0
        else:
            cand.ok = False
            cand.loss = 1.0
            cand.error = (proc.stderr or f"http_code={http_code}").strip()
        return cand

    def _score_all(self, candidates: list[PathCandidate]) -> None:
        ok = [c for c in candidates if c.ok and c.throughput_bps > 0]
        if not ok:
            for c in candidates:
                c.score = 0.0
            return

        max_tp = max(c.throughput_bps for c in ok)
        # Cap RTT normalization so extreme outliers don't dominate.
        rtts = [c.rtt_ms for c in ok if math.isfinite(c.rtt_ms)]
        max_rtt = max(rtts) if rtts else 1.0
        if max_rtt <= 0:
            max_rtt = 1.0

        for c in candidates:
            if not c.ok or c.throughput_bps <= 0:
                c.score = 0.0
                continue
            tp_norm = c.throughput_bps / max_tp
            if math.isfinite(c.rtt_ms):
                # Lower RTT is better → invert after normalizing.
                rtt_norm = 1.0 - min(c.rtt_ms / max_rtt, 1.0)
            else:
                rtt_norm = 0.0
            loss_norm = 1.0 - min(max(c.loss, 0.0), 1.0)
            c.score = (
                W_THROUGHPUT * tp_norm
                + W_RTT * rtt_norm
                + W_LOSS * loss_norm
            )


def _curl_route_args(kind: str, config: dict[str, Any]) -> list[str]:
    if kind == "wireguard":
        iface = config.get("interface")
        if not iface:
            return []
        return ["--interface", str(iface)]

    if kind == "proxy":
        ptype = str(config.get("type", "http")).lower()
        host = config["host"]
        port = int(config["port"])
        user = config.get("username")
        password = config.get("password")
        auth = ""
        if user is not None:
            auth = f"{user}:{password or ''}@"
        if ptype in ("socks5", "socks5h"):
            return ["--socks5-hostname", f"{auth}{host}:{port}"]
        if ptype == "socks4":
            return ["--socks4", f"{host}:{port}"]
        # http / https proxy
        return ["--proxy", f"http://{auth}{host}:{port}"]

    return []


# Module-level singleton for other project parts:
#   from pathp import path_prober
#   best = path_prober.best_path
path_prober = PathProber()


if __name__ == "__main__":
    best = path_prober.probe()
    print("candidates:")
    for c in path_prober.candidates:
        status = "ok" if c.ok else f"fail({c.error})"
        print(
            f"  {c.id:16}  score={c.score:.3f}  "
            f"tp={c.throughput_bps/1e6:.3f} MB/s  "
            f"rtt={c.rtt_ms:.1f} ms  loss={c.loss:.2f}  [{status}]"
        )
    if best is None:
        print("best_path: (none)")
    else:
        print(
            f"best_path: {best.id} ({best.kind}) score={best.score:.3f} "
            f"curl_args={best.curl_args}"
        )
