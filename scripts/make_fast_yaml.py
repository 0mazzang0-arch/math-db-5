#!/usr/bin/env python3
import re
from pathlib import Path

FAST_DISABLE_KEYS = {
    "use_table_recognition",
    "use_formula_recognition",
    "use_chart_recognition",
    "use_seal_recognition",
    "use_doc_unwarping",
    "use_doc_orientation_classify",
    "use_textline_orientation",
}


def replace_bool_line(line: str, key: str, value: bool) -> str:
    m = re.match(rf"^(\s*){re.escape(key)}\s*:\s*(true|false|True|False)(\s*(#.*)?)$", line)
    if not m:
        return line
    return f"{m.group(1)}{key}: {'true' if value else 'false'}{m.group(3) or ''}"


def main() -> int:
    cfg_dir = Path("configs")
    full_yaml = cfg_dir / "PP-StructureV3_full.yaml"
    fast_yaml = cfg_dir / "PP-StructureV3_fast.yaml"

    if not full_yaml.exists():
        print(f"missing: {full_yaml}")
        return 2

    lines = full_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines:
        updated = line
        for key in FAST_DISABLE_KEYS:
            updated = replace_bool_line(updated, key, False)
        updated = replace_bool_line(updated, "use_region_detection", True)
        updated = replace_bool_line(updated, "use_doc_preprocessor", True)
        if re.match(r"^\s*batch_size\s*:\s*\d+\s*(#.*)?$", updated):
            lead = re.match(r"^(\s*)", updated).group(1)  # type: ignore[union-attr]
            updated = f"{lead}batch_size: 1"
        out.append(updated)

    cfg_dir.mkdir(parents=True, exist_ok=True)
    fast_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"generated: {fast_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
