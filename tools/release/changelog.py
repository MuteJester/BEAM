"""Print the CHANGELOG.md section for one version; fail if there is none.

    python tools/release/changelog.py 1.0.0 > notes.md

The release workflow uses this both as a gate (no release without its notes)
and as the body of the GitHub Release.
"""
import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str:
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
                  text, re.S | re.M)
    return m.group(1).strip() if m else ""


if __name__ == "__main__":
    version = sys.argv[1]
    body = section(Path("CHANGELOG.md").read_text(), version)
    if not body:
        sys.exit(f"CHANGELOG.md has no '## [{version}]' section with content")
    print(body)
