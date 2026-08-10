#!/usr/bin/env python3
"""Lint every codebook in codebooks/ against SCHEMA.md.

Exit 0 when all files are usable (warnings are fine), 1 when any file has an error.

    python scripts/codebook/validate.py
    python scripts/codebook/validate.py --codebook violence
    python scripts/codebook/validate.py --strict        # warnings fail too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.codebook import load_all, load_codebook, render_prefix, prefix_hash  # noqa: E402
from lib.config import Config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codebook", help="lint only this codebook id")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--show-prompt", action="store_true", help="print the rendered prompt prefix")
    args = ap.parse_args()

    cfg = Config.load()
    cb_dir = cfg.path_for("codebooks")
    if not cb_dir.exists():
        print(f"codebook directory not found: {cb_dir}", file=sys.stderr)
        return 1

    if args.codebook:
        path = cb_dir / f"{args.codebook}.yaml"
        if not path.exists():
            print(f"no such codebook: {path}", file=sys.stderr)
            return 1
        results = [load_codebook(path)]
    else:
        results = load_all(cb_dir)

    if not results:
        print(f"no codebooks found in {cb_dir}")
        return 1

    n_err = n_warn = 0
    chars_per_token = float(cfg.get("limits.chars_per_token", 3.7))

    for res in results:
        name = res.path.name if res.path else "?"
        cb = res.codebook
        header = f"{name}"
        if cb:
            header += f"  (id={cb.id}, v{cb.version}, {len(cb.dimensions)} dimension(s))"
        print("=" * 78)
        print(header)

        for e in res.errors:
            print(f"  ERROR  {e}")
        for w in res.warnings:
            print(f"  warn   {w}")
        n_err += len(res.errors)
        n_warn += len(res.warnings)

        if cb:
            dims = ", ".join(
                f"{d.name}[{len(d.categories)}"
                + (", optional" if not d.required else "")
                + (f", if {d.conditional_on.dimension}" if d.conditional_on else "")
                + "]"
                for d in cb.dimensions
            )
            prefix = render_prefix(cb)
            print(f"  dimensions: {dims}")
            print(
                f"  prompt prefix: {len(prefix):,} chars "
                f"(~{len(prefix) / chars_per_token:,.0f} tokens), hash {prefix_hash(cb)}"
            )
            print(f"  matrix columns: {len(cb.code_columns())}")
            if args.show_prompt:
                print("-" * 78)
                print(prefix)
                print("-" * 78)
        if not res.errors and not res.warnings:
            print("  clean")

    print("=" * 78)
    print(f"{len(results)} codebook(s): {n_err} error(s), {n_warn} warning(s)")

    if n_err:
        print("\nFix the errors above. The pipeline refuses to run a codebook that fails to validate.")
        return 1
    if n_warn and args.strict:
        print("\n--strict: warnings treated as failures.")
        return 1
    if n_warn:
        print(
            "\nWarnings do not block a run, but negative_examples and an empty worked example "
            "are what keep prevalence honest. See codebooks/SCHEMA.md §5."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
