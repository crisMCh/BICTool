"""Logo sketches for BICTool: a CT gantry (the Q) with the patient table leaving the bore at an angle (the I / the
tail of the Q) - "IQ" for image quality. Writes one SVG per variant and a comparison sheet (dark + light background).

Run:  .venv/Scripts/python.exe design/logo/make_logo_sheet.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent
DARK_BG, LIGHT_BG = "#11151a", "#ffffff"
ACCENT = "#49c8de"           # BICTool accent (cyan)
INK_DARK, INK_LIGHT = "#e7eaed", "#1f2933"
C = (120, 112)               # gantry / bore centre in a 240 x 240 tile
BORE_R = 46
BED_ANGLE = 42               # degrees below horizontal
BED_W, BED_L = 26, 122


def gantry(ink, shape="rounded"):
    """Gantry housing with the circular bore cut out (even-odd fill)."""
    cx, cy = C
    if shape == "rounded":
        outer = (f"M {cx - 80} {cy - 24} Q {cx - 80} {cy - 80} {cx - 24} {cy - 80} L {cx + 24} {cy - 80} "
                 f"Q {cx + 80} {cy - 80} {cx + 80} {cy - 24} L {cx + 80} {cy + 24} Q {cx + 80} {cy + 80} {cx + 24} {cy + 80} "
                 f"L {cx - 24} {cy + 80} Q {cx - 80} {cy + 80} {cx - 80} {cy + 24} Z")
    else:
        outer = f"M {cx - 80} {cy} a 80 80 0 1 0 160 0 a 80 80 0 1 0 -160 0 Z"
    bore = f"M {cx - BORE_R} {cy} a {BORE_R} {BORE_R} 0 1 0 {2 * BORE_R} 0 a {BORE_R} {BORE_R} 0 1 0 {-2 * BORE_R} 0 Z"
    return f'<path d="{outer} {bore}" fill="{ink}" fill-rule="evenodd"/>'


def gantry_outline(ink, width=9):
    cx, cy = C
    return (f'<rect x="{cx - 76}" y="{cy - 76}" width="152" height="152" rx="58" fill="none" stroke="{ink}" '
            f'stroke-width="{width}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{BORE_R + 2}" fill="none" stroke="{ink}" stroke-width="{width}"/>')


def bed(fill, gap_colour, start=(-4, 0), gap=7, outline=None):
    """The table: a rounded bar starting in the bore and leaving it down-right (= the Q's tail / the I).
    `gap_colour` cuts the gantry where the bed crosses it, like the tail of a Q."""
    sx, sy = C[0] + start[0], C[1] + start[1]
    rect = f'x="0" y="{-BED_W / 2}" width="{BED_L}" height="{BED_W}" rx="{BED_W / 2}"'
    t = f'transform="translate({sx} {sy}) rotate({BED_ANGLE})"'
    cut = f'<rect {rect} {t} fill="{gap_colour}" stroke="{gap_colour}" stroke-width="{2 * gap}"/>'
    body = (f'<rect {rect} {t} fill="{fill}"/>' if outline is None else
            f'<rect {rect} {t} fill="{gap_colour}" stroke="{outline}" stroke-width="8"/>')
    return cut + body


def brain_axial(ink, cx, cy, s=1.0, width=3.2):
    """Simple axial brain seen from above: outline, the falx (midline) and a few sulci."""
    def p(x, y):
        return f"{cx + s * x:.1f} {cy + s * y:.1f}"
    outline = (f"M {p(0, -27)} C {p(13, -28)} {p(22, -18)} {p(22, -3)} C {p(22, 14)} {p(14, 26)} {p(0, 27)} "
               f"C {p(-14, 26)} {p(-22, 14)} {p(-22, -3)} C {p(-22, -18)} {p(-13, -28)} {p(0, -27)} Z")
    falx = f"M {p(0, -27)} L {p(0, 27)}"
    sulci = " ".join([
        f"M {p(-15, -14)} q {s * 5:.1f} {s * 3:.1f} {s * 2:.1f} {s * 9:.1f}",
        f"M {p(15, -14)} q {-s * 5:.1f} {s * 3:.1f} {-s * 2:.1f} {s * 9:.1f}",
        f"M {p(-17, 6)} q {s * 6:.1f} {-s * 1:.1f} {s * 7:.1f} {s * 7:.1f}",
        f"M {p(17, 6)} q {-s * 6:.1f} {-s * 1:.1f} {-s * 7:.1f} {s * 7:.1f}",
        f"M {p(-7, -22)} q {s * 1:.1f} {s * 5:.1f} {-s * 4:.1f} {s * 8:.1f}",
        f"M {p(7, -22)} q {-s * 1:.1f} {s * 5:.1f} {s * 4:.1f} {s * 8:.1f}",
    ])
    st = f'fill="none" stroke="{ink}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'
    return f'<path d="{outline}" {st}/><path d="{falx} {sulci}" {st}/>'


def ct_slice(cx, cy, bg):
    """A tiny axial head-CT slice in the bore: skull ring, grey brain, dark ventricles."""
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="27" ry="32" fill="#f4f6f8"/>'
            f'<ellipse cx="{cx}" cy="{cy}" rx="23" ry="28" fill="#8a949e"/>'
            f'<path d="M {cx - 2} {cy - 9} C {cx - 9} {cy - 10} {cx - 11} {cy - 1} {cx - 6} {cy + 6} '
            f'C {cx - 4} {cy + 2} {cx - 2} {cy - 2} {cx - 1} {cy - 3} Z" fill="{bg}"/>'
            f'<path d="M {cx + 2} {cy - 9} C {cx + 9} {cy - 10} {cx + 11} {cy - 1} {cx + 6} {cy + 6} '
            f'C {cx + 4} {cy + 2} {cx + 2} {cy - 2} {cx + 1} {cy - 3} Z" fill="{bg}"/>'
            f'<line x1="{cx}" y1="{cy - 28}" x2="{cx}" y2="{cy - 12}" stroke="#f4f6f8" stroke-width="1.6"/>')


def mtf_wave(cx, cy, ink):
    """A wave whose amplitude falls off - the MTF idea (sharpness), drawn across the bore."""
    import math
    pts = []
    for i in range(61):
        x = -30 + i
        amp = 12 * math.exp(-((x + 30) / 34) ** 2)
        pts.append(f"{cx + x:.1f},{cy + amp * math.sin((x + 30) * 0.42):.1f}")
    return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ink}" stroke-width="3.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def variant(name, bg, ink, accent):
    cx, cy = C
    if name == "A_plain":
        return gantry(ink) + bed(accent, bg)
    if name == "B_brain":
        return gantry(ink) + brain_axial(accent, cx - 8, cy - 9, 1.12) + bed(ink, bg, start=(10, 18))
    if name == "C_ctslice":
        return gantry(ink) + ct_slice(cx - 4, cy - 6, bg) + bed(accent, bg, start=(14, 24))
    if name == "D_outline":
        return gantry_outline(ink) + brain_axial(ink, cx - 8, cy - 9, 1.12, width=4) + bed(bg, bg, start=(10, 18),
                                                                                             outline=accent)
    if name == "E_round_wave":
        return gantry(ink, "circle") + mtf_wave(cx - 6, cy - 10, accent) + bed(accent, bg, start=(10, 18))
    raise ValueError(name)


NAMES = ["A_plain", "B_brain", "C_ctslice", "D_outline", "E_round_wave"]
LABELS = {"A_plain": "A  plain IQ", "B_brain": "B  brain in the bore", "C_ctslice": "C  head-CT slice in the bore",
          "D_outline": "D  line art", "E_round_wave": "E  round + sharpness wave"}


def tile(name, bg, ink, x, y):
    return f'<g transform="translate({x} {y})"><rect width="240" height="240" rx="18" fill="{bg}"/>' \
           f'{variant(name, bg, ink, ACCENT)}</g>'


def wordmark(name, bg, ink, x, y):
    return (f'<g transform="translate({x} {y})"><rect width="620" height="240" rx="18" fill="{bg}"/>'
            f'{variant(name, bg, ink, ACCENT)}'
            f'<text x="236" y="136" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="72" '
            f'font-weight="600" fill="{ink}" letter-spacing="1">BIC<tspan fill="{ACCENT}">Tool</tspan></text>'
            f'<text x="240" y="172" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="19" '
            f'fill="#8d96a0" letter-spacing="2.5">IMAGE QUALITY IN HEAD CT</text></g>')


def sheet():
    w, h = 40 + 5 * 260, 40 + 2 * 290 + 2 * 270
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="#e9edf1"/>']
    for row, (bg, ink) in enumerate(((DARK_BG, INK_DARK), (LIGHT_BG, INK_LIGHT))):
        for i, n in enumerate(NAMES):
            x, y = 20 + i * 260, 20 + row * 290
            parts.append(tile(n, bg, ink, x, y))
            parts.append(f'<text x="{x + 6}" y="{y + 262}" font-family="Segoe UI, Arial" font-size="16" '
                         f'fill="#1f2933">{LABELS[n]}</text>')
    parts.append(wordmark("B_brain", DARK_BG, INK_DARK, 20, 600))
    parts.append(wordmark("A_plain", LIGHT_BG, INK_LIGHT, 660, 600))
    parts.append(wordmark("C_ctslice", DARK_BG, INK_DARK, 20, 860))
    parts.append(wordmark("E_round_wave", LIGHT_BG, INK_LIGHT, 660, 860))
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    (OUT / "logo_sheet.svg").write_text(sheet(), encoding="utf-8")
    for n in NAMES:
        for tag, bg, ink in (("dark", DARK_BG, INK_DARK), ("light", LIGHT_BG, INK_LIGHT)):
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">'
                   f'{variant(n, bg, ink, ACCENT) if tag == "dark" else variant(n, bg, ink, ACCENT)}</svg>')
            svg = svg.replace('<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">',
                              f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">'
                              f'<rect width="240" height="240" rx="18" fill="{bg}"/>')
            (OUT / f"logo_{n}_{tag}.svg").write_text(svg, encoding="utf-8")
    print("written to", OUT)
