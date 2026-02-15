#!/usr/bin/env python3
from pathlib import Path


def main() -> int:
    out_dir = Path("configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "PP-StructureV3_full.yaml"

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3()
    pipeline.export_paddlex_config_to_yaml(str(out_path))
    print(f"exported: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
