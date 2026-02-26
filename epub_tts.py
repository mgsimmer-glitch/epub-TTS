#!/usr/bin/env python3
"""
epub-tts: Convert epub files to audiobooks using Microsoft Edge TTS.

Usage:
  epub-tts book.epub
  epub-tts book.epub -o ./my_audiobook
  epub-tts book.epub -v en-US-GuyNeural
  epub-tts --list-voices --locale en-US
"""

import argparse
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import edge_tts


# ---------------------------------------------------------------------------
# Epub parsing
# ---------------------------------------------------------------------------

def get_book_metadata(book: epub.EpubBook) -> dict:
    title = book.get_metadata("DC", "title")
    author = book.get_metadata("DC", "creator")
    return {
        "title": title[0][0] if title else "Unknown Title",
        "author": author[0][0] if author else "Unknown Author",
    }


def _build_toc_map(toc_items) -> dict:
    """Return a {filename: title} map from the epub TOC."""
    mapping = {}
    for item in toc_items:
        if isinstance(item, epub.Link):
            href = item.href.split("#")[0]
            mapping[href] = item.title
        elif isinstance(item, tuple):
            section, children = item
            if hasattr(section, "href"):
                href = section.href.split("#")[0]
                mapping[href] = section.title
            mapping.update(_build_toc_map(children))
    return mapping


def clean_text(html_content: bytes) -> str:
    """Strip HTML and normalise whitespace for TTS."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "head", "nav", "aside", "figure"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", " ", text)
    text = text.strip()
    return text


def extract_chapters(epub_path: str) -> tuple[dict, list[dict]]:
    """
    Extract chapters from epub in spine (reading) order.

    Returns:
        (metadata, chapters)  where each chapter is {"title": str, "text": str}
    """
    book = epub.read_epub(epub_path)
    metadata = get_book_metadata(book)
    toc_map = _build_toc_map(book.toc)

    chapters = []
    seen = set()

    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        file_name = item.get_name()
        if file_name in seen:
            continue
        seen.add(file_name)

        text = clean_text(item.get_content())
        if len(text) < 50:          # skip cover pages / empty sections
            continue

        # Determine chapter title: TOC > first heading > fallback
        title = toc_map.get(file_name)
        if not title:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            heading = soup.find(["h1", "h2", "h3"])
            title = heading.get_text(strip=True) if heading else f"Section {len(chapters) + 1}"

        chapters.append({"title": title, "text": text})

    return metadata, chapters


# ---------------------------------------------------------------------------
# Text chunking (edge-tts handles long texts, but very large chapters
# benefit from explicit splitting so progress is visible)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 4000   # characters


def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """Split text at sentence boundaries, staying under max_chars."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        slen = len(sentence)
        if current_len + slen + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [sentence], slen
        else:
            current.append(sentence)
            current_len += slen + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# TTS conversion
# ---------------------------------------------------------------------------

async def _tts_chunk(text: str, path: str, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


async def convert_chapter(
    chapter: dict,
    index: int,
    total: int,
    output_dir: Path,
    voice: str,
) -> Path:
    """Convert one chapter to an MP3 file."""
    chapter_num = f"{index:03d}"
    safe_title = re.sub(r"[^\w\s-]", "", chapter["title"])[:50].strip()
    safe_title = re.sub(r"\s+", "_", safe_title) or "chapter"
    output_file = output_dir / f"{chapter_num}_{safe_title}.mp3"

    print(f"  [{index}/{total}] {chapter['title']}  ({len(chapter['text'])} chars)")

    chunks = chunk_text(chapter["text"])

    if len(chunks) == 1:
        await _tts_chunk(chapter["text"], str(output_file), voice)
    else:
        # Convert each chunk to a temporary file, then concatenate raw bytes.
        # Raw MP3 concatenation works because each file starts with valid
        # MPEG frames; most players handle it correctly.
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths = []
            for ci, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk_{ci:04d}.mp3")
                print(f"     chunk {ci + 1}/{len(chunks)}")
                await _tts_chunk(chunk, chunk_path, voice)
                chunk_paths.append(chunk_path)

            with open(output_file, "wb") as out:
                for cp in chunk_paths:
                    with open(cp, "rb") as f:
                        out.write(f.read())

    return output_file


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_mp3s(chapter_paths: list[Path], output_path: Path) -> None:
    """Concatenate MP3 files into a single file, then delete the originals."""
    print(f"\nMerging {len(chapter_paths)} chapters → {output_path.name}")
    with open(output_path, "wb") as out:
        for path in chapter_paths:
            with open(path, "rb") as f:
                out.write(f.read())

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Merged file: {output_path}  ({size_mb:.1f} MB)")


def remove_chapter_files(chapter_paths: list[Path]) -> None:
    for path in chapter_paths:
        path.unlink(missing_ok=True)
    print(f"Removed {len(chapter_paths)} chapter file(s).")


async def convert_epub(
    epub_path: str,
    output_dir: str | None,
    voice: str,
    keep_chapters: bool = False,
    on_progress=None,   # async callable(event: str, **kwargs)
) -> Path:
    """Full epub → audiobook pipeline. Returns the path to the merged MP3."""
    epub_path = Path(epub_path)

    print(f"Parsing: {epub_path.name}")
    metadata, chapters = extract_chapters(str(epub_path))

    print(f"\nTitle:    {metadata['title']}")
    print(f"Author:   {metadata['author']}")
    print(f"Chapters: {len(chapters)}\n")

    if not chapters:
        raise ValueError("No readable chapters found. The epub may use an unusual structure.")

    if on_progress:
        await on_progress(
            "start",
            title=metadata["title"],
            author=metadata["author"],
            total_chapters=len(chapters),
        )

    # Resolve output directory
    if output_dir is None:
        safe = re.sub(r"[^\w\s-]", "", metadata["title"])[:40].strip()
        safe = re.sub(r"\s+", "_", safe) or "audiobook"
        output_dir = epub_path.parent / safe

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out}\n")

    print("Converting:")
    chapter_paths = []
    for i, chapter in enumerate(chapters, 1):
        if on_progress:
            await on_progress(
                "chapter_start",
                index=i,
                total=len(chapters),
                title=chapter["title"],
            )
        path = await convert_chapter(chapter, i, len(chapters), out, voice)
        chapter_paths.append(path)
        if on_progress:
            await on_progress("chapter_done", index=i, total=len(chapters))

    # Merge into a single audiobook file
    safe_title = re.sub(r"[^\w\s-]", "", metadata["title"])[:60].strip()
    safe_title = re.sub(r"\s+", "_", safe_title) or "audiobook"
    merged_path = out / f"{safe_title}.mp3"

    if on_progress:
        await on_progress("merging", total_chapters=len(chapter_paths))

    merge_mp3s(chapter_paths, merged_path)

    if not keep_chapters:
        remove_chapter_files(chapter_paths)
        try:
            out.rmdir()
        except OSError:
            pass

    print(f"\nDone. Audiobook saved to: {merged_path}")
    return merged_path


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------

async def list_voices_cmd(locale_filter: str | None) -> None:
    voices = await edge_tts.list_voices()
    fmt = "{:<42} {:<10} {}"
    print(fmt.format("Voice", "Locale", "Gender"))
    print("-" * 65)
    for v in sorted(voices, key=lambda x: x["Locale"]):
        if locale_filter and locale_filter.lower() not in v["Locale"].lower():
            continue
        print(fmt.format(v["ShortName"], v["Locale"], v["Gender"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="epub-tts",
        description="Convert an epub file to an audiobook using Microsoft Edge TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  epub-tts book.epub
  epub-tts book.epub -o ./output_dir
  epub-tts book.epub -v en-US-GuyNeural
  epub-tts book.epub --keep-chapters
  epub-tts --list-voices
  epub-tts --list-voices --locale en-GB
""",
    )
    parser.add_argument("epub_file", nargs="?", help="Path to the .epub file")
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: auto-named from book title)",
    )
    parser.add_argument(
        "-v", "--voice",
        default="en-US-AriaNeural",
        help="Edge TTS voice name (default: en-US-AriaNeural)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available Edge TTS voices and exit",
    )
    parser.add_argument(
        "--locale",
        help="Filter voices by locale when using --list-voices (e.g. en-US)",
    )
    parser.add_argument(
        "--keep-chapters",
        action="store_true",
        help="Keep individual chapter MP3 files alongside the merged audiobook",
    )

    args = parser.parse_args()

    if args.list_voices:
        asyncio.run(list_voices_cmd(args.locale))
        return

    if not args.epub_file:
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.epub_file):
        print(f"Error: file not found: {args.epub_file}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(convert_epub(args.epub_file, args.output, args.voice, args.keep_chapters))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
