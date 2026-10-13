"""Exercise the API and a separately running worker without paid model calls."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    token = os.environ["EVALMUT_SERVICE_TOKEN"]
    key = str(uuid.uuid4())

    def request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            args.url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result: dict[str, Any] = json.load(response)
            return result

    for attempt in range(60):
        try:
            request("GET", "/health/ready")
            break
        except (OSError, urllib.error.URLError):
            if attempt == 59:
                raise
            time.sleep(0.5)

    payload = {"split": "validation", "adapter": "deterministic"}
    run = request("POST", "/v1/runs", payload)
    repeated = request("POST", "/v1/runs", payload)
    assert run["run_id"] == repeated["run_id"]
    try:
        request("POST", "/v1/runs", {**payload, "split": "challenge"})
    except urllib.error.HTTPError as exc:
        assert exc.code == 409
    else:
        raise AssertionError("changed idempotent submission should return 409")

    for _attempt in range(60):
        result = request("GET", f"/v1/runs/{run['run_id']}")
        if result["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert result["status"] == "completed", result["error"]
    assert len(result["result"]["predictions"]) == 60
    assert result["result"]["errors"] == 0
    assert result["result"]["manifest"]["dataset_sha256"] == run["request"]["dataset_sha256"]
    assert result["result"]["metrics"]["n"] == 60
    print(
        json.dumps(
            {"run_id": run["run_id"], "status": "completed", "pairs": 60, "idempotency": "verified"}
        )
    )


if __name__ == "__main__":
    main()
