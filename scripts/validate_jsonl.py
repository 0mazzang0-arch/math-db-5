#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/validate_jsonl.py <path.jsonl>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}")
        return 2

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    failures = []
    for i, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            json.loads(raw)
        except Exception as e:
            failures.append((i, str(e)))

    if not failures:
        print(f"OK: {path} ({len(lines)} lines)")
        return 0

    print(f"FAIL: {path} invalid_json_lines={len(failures)}")
    for line_no, err in failures:
        print(f"- line {line_no}: {err}")
        start = max(1, line_no - 1)
        end = min(len(lines), line_no + 1)
        for idx in range(start, end + 1):
            prefix = ">" if idx == line_no else " "
            print(f"  {prefix} {idx}: {lines[idx-1]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
