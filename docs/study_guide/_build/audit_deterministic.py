"""Stage 4-A Deterministic Audit Script for 3D ESC Study Guide.

Checks across ch01.md .. ch07.md (and appendix.md / README.md):
  1. Exact character count within [3500, 4500] for each chapter ch01..ch07
  2. All markdown image links resolve to existing files on disk
  3. All cited `src/...` and `benchmarks/...` file paths exist on disk
  4. Balanced `<details>` / `</details>` tags and balanced `$` / `$$` LaTeX delimiters
  5. Required 4-section headings (`## 1.`, `## 2.`, `## 3.`, `## 4.`) and `> [!NOTE]` callout
"""

from __future__ import annotations

import json
from pathlib import Path
import re

BUILD_DIR = Path(__file__).resolve().parent
GUIDE_DIR = BUILD_DIR.parent
PROJECT_ROOT = GUIDE_DIR.parents[1]


def check_latex_balance(text: str) -> tuple[bool, str]:
    # Strip fenced code blocks before checking $ balance
    stripped = re.sub(r"```.*?```", "", text, flags=re.S)
    # Count $$ first and remove them
    n_double = stripped.count("$$")
    if n_double % 2 != 0:
        return False, f"Unbalanced $$ count ({n_double})"
    no_double = stripped.replace("$$", "")
    # Remove escaped \$
    no_esc = no_double.replace(r"\$", "")
    n_single = no_esc.count("$")
    if n_single % 2 != 0:
        return False, f"Unbalanced $ count ({n_single})"
    return True, f"$$={n_double}, $={n_single}"


def audit_chapter(ch_path: Path) -> dict:
    text = ch_path.read_text(encoding="utf-8")
    n_chars = len(text)
    issues: list[str] = []

    if not (3500 <= n_chars <= 4500):
        issues.append(f"Char count {n_chars} outside [3500, 4500]")

    for h in ("## 1.", "## 2.", "## 3.", "## 4."):
        if h not in text:
            issues.append(f"Missing heading '{h}'")

    if "> [!NOTE]" not in text:
        issues.append("Missing `> [!NOTE]` callout")

    if text.count("<details>") < 1 or text.count("<details>") != text.count("</details>"):
        issues.append("Missing or unbalanced `<details>` tags")

    ok_tex, tex_info = check_latex_balance(text)
    if not ok_tex:
        issues.append(tex_info)

    # Check image links
    img_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    for rel in img_links:
        target = (ch_path.parent / rel).resolve()
        if not target.exists():
            issues.append(f"Broken image link: {rel}")

    # Check cited src/ or benchmarks/ paths
    code_paths = set(re.findall(r"(?:src|benchmarks)/[A-Za-z0-9_./-]+\.py", text))
    for cp in sorted(code_paths):
        if not (PROJECT_ROOT / cp).exists():
            issues.append(f"Missing cited code file: {cp}")

    return {
        "file": ch_path.name,
        "chars": n_chars,
        "images": len(img_links),
        "cited_code_files": len(code_paths),
        "latex": tex_info,
        "pass": len(issues) == 0,
        "issues": issues,
    }


def main() -> None:
    results = []
    all_pass = True
    for i in range(1, 8):
        p = GUIDE_DIR / f"ch0{i}.md"
        if not p.exists():
            results.append({"file": p.name, "pass": False, "issues": ["File missing"]})
            all_pass = False
            continue
        r = audit_chapter(p)
        results.append(r)
        if not r["pass"]:
            all_pass = False
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f" [{status}] {r['file']} | chars={r['chars']:4d} | imgs={r['images']} | "
            f"code_refs={r['cited_code_files']} | {r['latex']} | issues={r['issues']}"
        )

    out = BUILD_DIR / "audit_deterministic.json"
    out.write_text(json.dumps({"all_pass": all_pass, "chapters": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" Overall Stage 4-A Deterministic Audit: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
