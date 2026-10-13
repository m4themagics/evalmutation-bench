"""Run retail acceptance against an isolated local Compose stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--restart-worker", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/retail-session.json"))
    args = parser.parse_args()
    token = os.environ["EVALMUT_SERVICE_TOKEN"]

    def request(
        method: str, path: str, payload: dict[str, Any] | None = None, key: str | None = None
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        req = urllib.request.Request(
            args.url + path,
            method=method,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result: dict[str, Any] = json.load(response)
            return result

    def wait_run(run_id: str) -> dict[str, Any]:
        for _ in range(60):
            run = request("GET", f"/v1/runs/{run_id}")
            if run["status"] in ("completed", "failed"):
                assert run["status"] == "completed", run["error"]
                return run
            time.sleep(0.5)
        raise TimeoutError("worker did not complete the command")

    for attempt in range(60):
        try:
            request("GET", "/health/ready")
            break
        except (OSError, urllib.error.URLError):
            if attempt == 59:
                raise
            time.sleep(0.5)

    payload = {"sku": "COFFEE-001", "target_days": 14}
    key = str(uuid.uuid4())
    started = request("POST", "/v1/retail/sessions", payload, key)
    assert request("POST", "/v1/retail/sessions", payload, key)["run_id"] == started["run_id"]
    wait_run(started["run_id"])
    sid = started["request"]["session_id"]
    pending = request("GET", f"/v1/retail/sessions/{sid}")
    assert pending["phase"] == "awaiting_approval" and pending["order"] is None
    assert pending["proposal"]["quantity"] == 120

    if args.restart_worker:
        subprocess.run(["docker", "compose", "restart", "worker"], check=True)
        restored = request("GET", f"/v1/retail/sessions/{sid}")
        assert restored["proposal_sha256"] == pending["proposal_sha256"]
        assert restored["phase"] == "awaiting_approval"

    approval = {"approved": True, "proposal_sha256": pending["proposal_sha256"]}
    key = str(uuid.uuid4())
    confirmed = request("POST", f"/v1/retail/sessions/{sid}/approval", approval, key)
    repeated = request("POST", f"/v1/retail/sessions/{sid}/approval", approval, key)
    assert confirmed["run_id"] == repeated["run_id"]
    wait_run(confirmed["run_id"])
    completed = request("GET", f"/v1/retail/sessions/{sid}")
    assert completed["phase"] == "ordered"
    assert completed["order"]["quantity"] == 120
    assert completed["order"]["total_cost_cents"] == 60000
    assert completed["trace"]["trace_id"] == sid
    try:
        request("POST", f"/v1/retail/sessions/{sid}/approval", approval, str(uuid.uuid4()))
    except urllib.error.HTTPError as exc:
        assert exc.code == 409
    else:
        raise AssertionError("second approval must not enqueue a second purchase")
    unchanged = request("GET", f"/v1/retail/sessions/{sid}")
    assert unchanged["order"]["order_id"] == completed["order"]["order_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "session_id": sid,
                "phase": completed["phase"],
                "order_id": completed["order"]["order_id"],
                "worker_restarted": args.restart_worker,
                "quantity": 120,
                "total_cost_cents": 60000,
            }
        )
    )


if __name__ == "__main__":
    main()
