#!/usr/bin/env python3
"""
Minimal Python client for the GLM-OCR FastAPI service.

Examples:
  # health check
  python scripts/client.py health

  # parse one or more local files
  python scripts/client.py parse invoice.pdf table.png --task table

  # parse by URL
  python scripts/client.py url https://example.com/page.png --task text

  # custom prompt
  python scripts/client.py parse form.png --task custom --prompt "Extract all phone numbers."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def _print(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_health(args: argparse.Namespace) -> int:
    r = httpx.get(f"{args.base_url}/health", timeout=10)
    _print(r.json())
    return 0 if r.json().get("ready") else 1


def cmd_parse(args: argparse.Namespace) -> int:
    files = []
    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"file not found: {p}", file=sys.stderr)
            return 2
        mime = "application/pdf" if p.suffix.lower() == ".pdf" else "image/png"
        files.append(("files", (p.name, p.read_bytes(), mime)))

    data = {"task": args.task}
    if args.prompt:
        data["prompt"] = args.prompt
    if args.max_new_tokens:
        data["max_new_tokens"] = str(args.max_new_tokens)

    r = httpx.post(
        f"{args.base_url}/ocr/parse",
        files=files,
        data=data,
        timeout=args.timeout,
    )
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    _print(r.json())
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    payload = {"images": args.urls, "task": args.task}
    if args.prompt:
        payload["prompt"] = args.prompt
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens

    r = httpx.post(
        f"{args.base_url}/ocr/parse/url",
        json=payload,
        timeout=args.timeout,
    )
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    _print(r.json())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--timeout", type=float, default=600.0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health")

    p_parse = sub.add_parser("parse", help="upload local files")
    p_parse.add_argument("files", nargs="+")
    p_parse.add_argument("--task", default="text", choices=["text", "formula", "table", "custom"])
    p_parse.add_argument("--prompt", default=None)
    p_parse.add_argument("--max-new-tokens", type=int, default=None)

    p_url = sub.add_parser("url", help="parse remote URLs")
    p_url.add_argument("urls", nargs="+")
    p_url.add_argument("--task", default="text", choices=["text", "formula", "table", "custom"])
    p_url.add_argument("--prompt", default=None)
    p_url.add_argument("--max-new-tokens", type=int, default=None)

    args = parser.parse_args()
    return {"health": cmd_health, "parse": cmd_parse, "url": cmd_url}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
