#!/usr/bin/env python3
"""Build paper-style LaTeX sources and PDFs for every Jekyll post."""

from __future__ import annotations

import re
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "post-sources"
HEADER = ROOT / "tools" / "post-pdf-header.tex"
BOX_FILTER = ROOT / "tools" / "post-pdf-boxes.lua"


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, flags=re.S)
    if not match:
        raise ValueError("Missing YAML front matter")
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        item = re.match(r"([A-Za-z_][\w-]*):\s*(.*)", line)
        if item:
            metadata[item.group(1)] = item.group(2).strip().strip('"')
    return metadata, match.group(2)


def paper_date(raw: str) -> str:
    try:
        parsed = datetime.strptime(raw[:10], "%Y-%m-%d")
        return parsed.strftime("%B %-d, %Y")
    except ValueError:
        return raw[:10]


def convert_liquid_boxes(body: str) -> str:
    pattern = re.compile(
        r"\{% capture (?P<variable>\w+) %\}\n"
        r"(?P<content>.*?)\n"
        r"\{% endcapture %\}\n"
        r"\{% include thms/(?P<kind>theorem|lemma|proof)\.html(?P<args>.*?)%\}",
        flags=re.S,
    )

    def replacement(match: re.Match[str]) -> str:
        kind = match.group("kind")
        content = match.group("content").strip()
        args = match.group("args")
        title_match = re.search(r'title="([^"]+)"', args)
        title = title_match.group(1) if title_match else ""
        if kind == "theorem":
            return f'::: {{.theorem-box title="{title}"}}\n\n{content}\n\n:::'
        if kind == "lemma":
            return f'::: {{.lemma-box title="{title}"}}\n\n{content}\n\n:::'
        return f"::: {{.proof-box}}\n\n{content}\n\n:::"

    converted = pattern.sub(replacement, body)
    if "{%" in converted or "{{" in converted:
        raise ValueError("Unsupported Liquid markup remains after preprocessing")
    return converted


def image_width(path: str, explicit_px: str | None) -> int:
    if explicit_px:
        pixels = int(explicit_px)
        return max(38, min(90, round(pixels / 8)))
    name = Path(path).name
    compact = {
        "sigmoid.png",
        "tanh.png",
        "hard-tanh.png",
        "softsign.png",
        "relu.png",
        "leaky-relu.png",
        "transformer-decoder.png",
        "transformer-encoder.png",
    }
    return 48 if name in compact else 76


def normalize_images(body: str) -> str:
    image_pattern = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\((?P<path>/assets/[^)]+)\)"
        r"(?:\{: *width=\"(?P<width>\d+)\" *\})?"
    )

    def replacement(match: re.Match[str]) -> str:
        path = match.group("path").lstrip("/")
        width = image_width(path, match.group("width"))
        return f"![{match.group('alt')}]({path}){{ width={width}% }}"

    return image_pattern.sub(replacement, body)


def normalize_heading_levels(body: str) -> str:
    levels = [
        len(match.group(1))
        for match in re.finditer(r"^(#{1,6})\s+", body, flags=re.M)
    ]
    if not levels:
        return body
    shift = min(levels) - 1
    if shift <= 0:
        return body

    def replacement(match: re.Match[str]) -> str:
        return "#" * (len(match.group(1)) - shift) + " "

    return re.sub(r"^(#{1,6})\s+", replacement, body, flags=re.M)


def normalize_manual_heading_numbers(body: str) -> str:
    """Use LaTeX numbering without duplicating numbers already in headings."""
    body = re.sub(
        r"^(#{1,6}\s+)\d+(?:\.\d+)+\s+",
        r"\1",
        body,
        flags=re.M,
    )
    return re.sub(
        r"^(#{1,6})[ \t]+(\([A-Za-zivxlcdmIVXLCDM]+\)(?:[ \t]+[^\n]*)?|\d{4})[ \t]*$",
        r"\1 \2 {.unnumbered}",
        body,
        flags=re.M,
    )


def preprocess_post(post: Path) -> tuple[dict[str, str], str]:
    metadata, body = split_front_matter(post.read_text(encoding="utf-8"))
    body = re.sub(
        r"(?m)(?:^>[^\n]*\n)+^\{: \.chinese-version-link \}\s*\n?",
        "",
        body,
        count=1,
    )
    body = re.sub(
        r"(?m)^[^\n]+\n^\{: \.web-only \}\s*\n?",
        "",
        body,
    )
    body = re.sub(r"^\{: \.chinese-version-link \}\s*$", "", body, flags=re.M)
    body = convert_liquid_boxes(body)
    body = normalize_images(body)
    body = normalize_manual_heading_numbers(body)
    body = normalize_heading_levels(body)
    body = body.replace("\u2011", "-").replace("\u2013", "--").replace("\u2014", "---")
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    return metadata, body


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    cache_dir = ROOT / "tmp" / "pdfs" / "tectonic-cache"
    environment["XDG_CACHE_HOME"] = str(cache_dir)
    environment["TECTONIC_CACHE_DIR"] = str(cache_dir)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def build_post(post: Path) -> tuple[Path, Path]:
    metadata, body = preprocess_post(post)
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", post.stem)
    markdown_path = TMP_DIR / f"{slug}.md"
    tex_path = OUTPUT_DIR / f"{slug}.tex"
    pdf_path = OUTPUT_DIR / f"{slug}.pdf"
    markdown_path.write_text(body, encoding="utf-8")

    run(
        [
            "pandoc",
            str(markdown_path),
            "--from=markdown+tex_math_dollars+raw_tex+pipe_tables+fenced_divs",
            "--to=latex",
            "--standalone",
            "--number-sections",
            "--top-level-division=section",
            f"--resource-path={ROOT}",
            f"--include-in-header={HEADER}",
            f"--lua-filter={BOX_FILTER}",
            "--metadata",
            f"title={metadata['title']}",
            "--metadata",
            "author=Selene",
            "--metadata",
            f"date={paper_date(metadata.get('date', ''))}",
            "--variable=documentclass:article",
            "--variable=classoption:a4paper",
            "--variable=classoption:10pt",
            "--variable=mainfont:Times New Roman",
            "--variable=mathfont:STIX Two Math",
            "--variable=CJKmainfont:Songti SC",
            "--variable=monofont:Menlo",
            "--variable=indent:true",
            "--variable=colorlinks:true",
            "--variable=linkcolor:black",
            "--variable=urlcolor:black",
            "--output",
            str(tex_path),
        ]
    )

    if slug == "neural-network-architectures-and-properties":
        tex = tex_path.read_text(encoding="utf-8")
        tex = tex.replace("\\begin{figure}\n", "\\begin{figure}[H]\n")
        tex_path.write_text(tex, encoding="utf-8")

    run(
        [
            "tectonic",
            "--keep-logs",
            "--outdir",
            str(OUTPUT_DIR),
            str(tex_path),
        ]
    )
    if not pdf_path.exists():
        raise RuntimeError(f"PDF was not created: {pdf_path}")
    return tex_path, pdf_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    posts = sorted(POSTS_DIR.glob("*.md"))
    if not posts:
        raise RuntimeError("No posts found")
    for post in posts:
        tex_path, pdf_path = build_post(post)
        print(f"built {tex_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
