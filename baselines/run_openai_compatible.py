from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline prompts through an OpenAI-compatible VLM API.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--variants", default="", help="Comma-separated baseline variants; empty means all.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install the openai package before running API inference.") from error

    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    rows = _read_jsonl(args.prompts)
    if selected_variants:
        rows = [row for row in rows if row.get("variant") in selected_variants]
    if args.limit > 0:
        rows = rows[: args.limit]

    output_path = Path(args.out)
    completed = {
        str(row.get("row_id") or "")
        for row in _read_jsonl(output_path)
    } if args.resume and output_path.exists() else set()
    results = _read_jsonl(output_path) if completed else []
    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url or None)

    for index, row in enumerate(rows, 1):
        if str(row.get("row_id") or "") in completed:
            continue
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_url(str(row.get("image") or ""))},
                            },
                            {"type": "text", "text": str(row.get("prompt") or "")},
                        ],
                    }
                ],
            )
            prediction = str(response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            error = ""
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
        except Exception as exception:
            prediction = ""
            error = repr(exception)
            tokens = 0
        results.append(
            {
                **row,
                "prediction": prediction,
                "prediction_source": args.model,
                "inference_seconds": round(time.perf_counter() - started, 4),
                "tokens": tokens,
                "error": error,
            }
        )
        write_jsonl(output_path, results)
        print(f"inference progress: {index}/{len(rows)}", flush=True)


def _image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    main()
