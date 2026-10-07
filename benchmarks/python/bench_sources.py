"""
What the benchmark sources and the library sources actually say, read at
report time rather than recorded in a hand-maintained map that rots.

- public_symbols(): every beam_* function declared in include/beam/*.h
- definitions():    where each beam_* function is defined, file:line
- benchmarks():     each BEAM_BENCHMARK, its location, and the beam_* calls
                    inside its timed block (what its time is the time OF)
- coverage():       which public functions some benchmark times
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_CALL = re.compile(r"\b(beam_[A-Za-z0-9_]+)\s*\(")
_DECL = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \t\*]*?\b(beam_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
_BENCH = re.compile(r"BEAM_BENCHMARK(?:_EX)?\(\s*(\w+)\s*,\s*(\w+)\s*,", re.MULTILINE)


def _strip_comments(text: str) -> str:
    """Blank out comments and string literals, keeping offsets and newlines."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", blank, text)
    text = re.sub(r'"(?:\\.|[^"\\\n])*"', blank, text)
    return text


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _match_brace(text: str, open_idx: int) -> int:
    """Index just past the brace that closes the one at open_idx."""
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


@lru_cache(maxsize=None)
def public_symbols(repo: Path = REPO_ROOT) -> Dict[str, str]:
    """beam_* functions declared in the public headers -> header path (repo-relative)."""
    out: Dict[str, str] = {}
    for h in sorted((repo / "include" / "beam").glob("*.h")):
        text = _strip_comments(h.read_text(encoding="utf-8"))
        for m in _DECL.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            if text[line_start:m.start()].lstrip().startswith("#"):
                continue  # a macro
            out.setdefault(m.group(1), str(h.relative_to(repo)))
    return out


@lru_cache(maxsize=None)
def definitions(repo: Path = REPO_ROOT) -> Dict[str, Tuple[str, int]]:
    """beam_* function definitions (a body follows the parameter list) -> (file, line)."""
    out: Dict[str, Tuple[str, int]] = {}
    files = list((repo / "src" / "native").rglob("*.c")) + list((repo / "include" / "beam").glob("*.h"))
    for f in sorted(files):
        raw = f.read_text(encoding="utf-8", errors="replace")
        text = _strip_comments(raw)
        for m in _DECL.finditer(text):
            # Walk the balanced parameter list, then look at what follows it.
            i, depth = m.end() - 1, 0
            while i < len(text):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            rest = text[i + 1:i + 200].lstrip()
            if rest.startswith("{"):
                out.setdefault(m.group(1), (str(f.relative_to(repo)), _line_of(text, m.start())))
    return out


@dataclass
class BenchSource:
    domain: str
    name: str
    file: str
    line: int
    timed_calls: List[str] = field(default_factory=list)   # beam_* inside the timed block
    setup_calls: List[str] = field(default_factory=list)   # beam_* outside it

    @property
    def key(self) -> str:
        return f"{self.domain}::{self.name}"


@lru_cache(maxsize=None)
def benchmarks(repo: Path = REPO_ROOT) -> Dict[str, BenchSource]:
    out: Dict[str, BenchSource] = {}
    for f in sorted((repo / "benchmarks" / "c" / "suites").glob("*.c")):
        text = _strip_comments(f.read_text(encoding="utf-8"))
        for m in _BENCH.finditer(text):
            # The macro's argument list closes, then the function body opens.
            body_open = text.find("{", text.find(")", m.end()))
            if body_open < 0:
                continue
            body_end = _match_brace(text, body_open)
            body = text[body_open:body_end]
            timed: List[str] = []
            tb = body.find("BEAM_TIMED_BLOCK")
            setup_text = body
            if tb >= 0:
                blk_open = body.find("{", tb)
                blk_end = _match_brace(body, blk_open)
                timed = _unique(_CALL.findall(body[blk_open:blk_end]))
                setup_text = body[:tb] + body[blk_end:]
            src = BenchSource(
                domain=m.group(1), name=m.group(2),
                file=str(f.relative_to(repo)), line=_line_of(text, m.start()),
                timed_calls=timed,
                setup_calls=[c for c in _unique(_CALL.findall(setup_text)) if c not in timed],
            )
            out[src.key] = src
    return out


def _unique(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    return [x for x in xs if not (x in seen or seen.add(x))]


def coverage(repo: Path = REPO_ROOT) -> Dict[str, object]:
    """Public functions timed by at least one benchmark, read from the sources."""
    pub = public_symbols(repo)
    timed_by: Dict[str, List[str]] = {}
    for b in benchmarks(repo).values():
        for c in b.timed_calls:
            timed_by.setdefault(c, []).append(b.key)
    covered = sorted(s for s in pub if s in timed_by)
    uncovered = sorted(s for s in pub if s not in timed_by)
    by_header: Dict[str, Dict[str, int]] = {}
    for s, h in pub.items():
        e = by_header.setdefault(h, {"total": 0, "covered": 0})
        e["total"] += 1
        e["covered"] += s in timed_by
    return {
        "total": len(pub),
        "covered": len(covered),
        "percent": (100.0 * len(covered) / len(pub)) if pub else 0.0,
        "uncovered": [{"symbol": s, "header": pub[s]} for s in uncovered],
        "by_header": by_header,
        "timed_by": {s: timed_by[s] for s in covered},
    }


def locate(symbol: str, repo: Path = REPO_ROOT) -> Optional[Tuple[str, int]]:
    return definitions(repo).get(symbol)
