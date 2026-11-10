"""Generate the app icons from code, with no image dependency.

An installed web app needs real PNG icons at 192 and 512 pixels; Chrome will not
offer installation without them, and a link to an SVG is not accepted on every
platform. Rather than commit opaque binaries with no way to reproduce them, the
mark is drawn here with the standard library and written out.

The mark is the product's own and uses the same three materials as the page: a
sheet of warm ink, the paper disc that is the microphone, and the single amber
lamp that means "live". Nothing here is a colour the interface does not already
use, so the installed icon and the open app are recognisably one thing. It is
drawn at 4x and box-filtered down, which is enough anti-aliasing for a solid
ring.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "icons"

BG = (20, 16, 13)            # #14100d  warm ink, the page's own ground
RING = (242, 234, 220)       # #f2eadc  paper
DOT = (217, 163, 97)         # #d9a361  the lamp: live
SUPERSAMPLE = 4


def _rounded_rect(x: float, y: float, size: float, radius: float) -> bool:
    """Point-in-rounded-square test, in icon coordinates."""
    inner = size - 2 * radius
    if radius <= x <= size - radius and 0 <= y <= size:
        return True
    if 0 <= x <= size and radius <= y <= size - radius:
        return True
    for cx, cy in ((radius, radius), (size - radius, radius), (radius, size - radius), (size - radius, size - radius)):
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            return True
    return False


def _pixel(x: float, y: float, size: float, pad: float) -> tuple[int, int, int, int]:
    """Colour of one point, in a design box of `size` inset by `pad`."""
    work = size - 2 * pad
    if work <= 0:
        return (*BG, 0)
    # Map into the padded working box.
    px, py = x - pad, y - pad
    if not (0 <= px <= work and 0 <= py <= work):
        return (*BG, 0)
    if not _rounded_rect(px, py, work, work * 0.23):
        return (*BG, 0)

    # Flat ink. A gradient on a launcher icon is a font-free way of looking
    # generic; the depth here comes from the two drawn objects.

    cx = cy = work / 2
    dx, dy = px - cx, py - cy
    distance = math.hypot(dx, dy)
    ring_radius = work * 0.30
    ring_width = work * 0.075
    if abs(distance - ring_radius) <= ring_width / 2:
        return (*RING, 255)

    # The lamp sits at the centre of the paper ring, the way it does on the
    # open microphone in the app; off-centre it read as a smudge on the edge.
    dot_x, dot_y, dot_r = work * 0.5, work * 0.5, work * 0.108
    if math.hypot(px - dot_x, py - dot_y) <= dot_r:
        return (*DOT, 255)

    return (*BG, 255)


def _render(size: int, pad_ratio: float = 0.0) -> bytes:
    """Return raw RGBA rows for one icon."""
    pad = size * pad_ratio
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            r = g = b = a = 0
            for sy in range(SUPERSAMPLE):
                for sx in range(SUPERSAMPLE):
                    px = (x * SUPERSAMPLE + sx + 0.5) / SUPERSAMPLE
                    py = (y * SUPERSAMPLE + sy + 0.5) / SUPERSAMPLE
                    pr, pg, pb, pa = _pixel(px, py, size, pad)
                    r += pr * pa
                    g += pg * pa
                    b += pb * pa
                    a += pa
            count = SUPERSAMPLE * SUPERSAMPLE
            if a:
                row += bytes((r // a, g // a, b // a, a // count))
            else:
                row += b"\x00\x00\x00\x00"
        rows.append(bytes(row))
    raw = b"".join(b"\x00" + row for row in rows)
    return raw


def _png(raw: bytes, size: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    targets = (
        ("icon-192.png", 192, 0.0),
        ("icon-512.png", 512, 0.0),
        # Maskable icons must keep their content inside the safe circle, so the
        # mark is inset and the background reaches the edges.
        ("icon-maskable-512.png", 512, 0.14),
        ("apple-touch-icon.png", 180, 0.0),
    )
    for name, size, pad in targets:
        path = OUT / name
        path.write_bytes(_png(_render(size, pad), size))
        written.append(path)
    (ROOT / "static" / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="15" fill="#14100d"/>'
        '<circle cx="32" cy="32" r="19" fill="none" stroke="#f2eadc" stroke-width="4.8"/>'
        '<circle cx="32" cy="32" r="6.9" fill="#d9a361"/>'
        "</svg>\n",
        encoding="utf-8",
    )
    return written


if __name__ == "__main__":
    for path in build():
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
