from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish public benchmark splits to HF Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, owner/name")
    parser.add_argument("--execute", action="store_true", help="Perform upload; default is dry-run")
    args = parser.parse_args()
    files = [Path("data/development.jsonl"), Path("data/validation.jsonl")]
    for path in files:
        if not path.exists():
            raise SystemExit(f"missing generated split: {path}")
    if not args.execute:
        print(f"dry-run: would upload {', '.join(map(str, files))} to {args.repo_id}")
        return
    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required with --execute")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub before publishing") from exc
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    for path in files:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print(f"published public splits to {args.repo_id}")


if __name__ == "__main__":
    main()
