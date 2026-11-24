"""Stage 0 Pre-flight Verification & Stage 1 AST Context Packet Extractor.

Verifies every file path, figure path, and AST symbol in `manifest.yaml`,
and extracts concise (<= 15 lines) ground-truth code snippets with exact
start-end line numbers into `context_ch01.json` .. `context_ch07.json`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

BUILD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUILD_DIR.parents[2]
MANIFEST_PATH = BUILD_DIR / "manifest.yaml"


def parse_manifest_simple(text: str) -> list[dict]:
    """Minimal dependency-free parser for our structured manifest.yaml."""
    chapters: list[dict] = []
    cur_ch: dict | None = None
    cur_list: str | None = None
    cur_item: dict | None = None

    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()

        if s.startswith("- id:"):
            cur_ch = {
                "id": s.split(":", 1)[1].strip().strip('"'),
                "title": "",
                "code_sources": [],
                "figures": [],
                "locked_metrics": [],
                "required_intuitions": [],
            }
            chapters.append(cur_ch)
            cur_list = None
            cur_item = None
        elif cur_ch is not None:
            if indent == 4 and s.startswith("title:"):
                cur_ch["title"] = s.split(":", 1)[1].strip().strip('"')
            elif indent == 4 and s.endswith(":"):
                cur_list = s[:-1]
                cur_item = None
            elif indent == 4 and s.endswith(": []"):
                cur_list = None
                cur_item = None
            elif cur_list in ("code_sources", "figures"):
                if s.startswith("- path:"):
                    cur_item = {"path": s.split(":", 1)[1].strip().strip('"')}
                    cur_ch[cur_list].append(cur_item)
                elif cur_item is not None and ":" in s:
                    k, v = s.split(":", 1)
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        cur_item[k.strip()] = [
                            x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()
                        ]
                    else:
                        cur_item[k.strip()] = v.strip('"')
            elif cur_list in ("locked_metrics", "required_intuitions") and s.startswith("- "):
                cur_ch[cur_list].append(s[2:].strip().strip('"'))
    return chapters


def find_symbol_snippet(py_path: Path, symbol: str, max_lines: int = 15) -> dict:
    src_lines = py_path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(src_lines))
    parts = symbol.split(".")
    target_node = None
    if len(parts) == 1:
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == parts[0]:
                target_node = n
                break
    elif len(parts) == 2:
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == parts[0]:
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and m.name == parts[1]:
                        target_node = m
                        break
    if target_node is None:
        raise ValueError(f"Symbol '{symbol}' NOT found in {py_path}")

    start = target_node.lineno
    end = min(target_node.end_lineno, start + max_lines - 1)
    snippet = "\n".join(src_lines[start - 1 : end])
    return {
        "symbol": symbol,
        "full_line_range": [target_node.lineno, target_node.end_lineno],
        "snippet_line_range": [start, end],
        "snippet": snippet,
    }


def main() -> None:
    chapters = parse_manifest_simple(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(chapters) == 7, f"Expected 7 chapters, got {len(chapters)}"

    for ch in chapters:
        packet = {
            "id": ch["id"],
            "title": ch["title"],
            "locked_metrics": ch["locked_metrics"],
            "required_intuitions": ch["required_intuitions"],
            "figures": [],
            "code_snippets": [],
        }
        # Verify figures exist on disk
        for fig in ch["figures"]:
            fpath = PROJECT_ROOT / fig["path"]
            if not fpath.exists():
                raise FileNotFoundError(f"Missing figure: {fpath}")
            packet["figures"].append({
                "path": fig["path"],
                "rel_from_study_guide": "../../" + fig["path"],
                "must_verify_value": fig.get("must_verify_value", ""),
                "exists": True,
            })
        # Verify code files & extract AST snippets
        for src in ch["code_sources"]:
            spath = PROJECT_ROOT / src["path"]
            if not spath.exists():
                raise FileNotFoundError(f"Missing source: {spath}")
            for sym in src.get("focus_symbols", []):
                info = find_symbol_snippet(spath, sym, max_lines=15)
                info["path"] = src["path"]
                packet["code_snippets"].append(info)

        out_path = BUILD_DIR / f"context_{ch['id']}.json"
        out_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f" [OK] {ch['id']} | {len(packet['code_snippets'])} code snippets | "
            f"{len(packet['figures'])} figures -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
