"""Logo sketches, round 2: a bigger, recognisable CT table as the I / the tail of the Q.

CT-table cues used: a long thin couch with visible thickness (two tones), a head cradle at the gantry end,
a column on a floor plate, a patient lying on it with the head (brain) in the bore.
Writes logo2_sheet.svg (dark + light) and one SVG per variant.

Run:  .venv/Scripts/python.exe design/logo/make_logo_sheet2.py
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
DARK_BG, LIGHT_BG = "#11151a", "#ffffff"
ACCENT, ACCENT_DARK = "#49c8de", "#2a8fa3"
INK_DARK, INK_LIGHT = "#e7eaed", "#1f2933"
SIZE = 260


def ring(ink, cx, cy, outer, bore, shape):
    if shape == "circle":
        o = f"M {cx - outer} {cy} a {outer} {outer} 0 1 0 {2 * outer} 0 a {outer} {outer} 0 1 0 {-2 * outer} 0 Z"
    else:  # rounded square housing, like a real gantry seen from the front
        k = outer * 0.32
        o = (f"M {cx - outer} {cy - k} Q {cx - outer} {cy - outer} {cx - k} {cy - outer} L {cx + k} {cy - outer} "
             f"Q {cx + outer} {cy - outer} {cx + outer} {cy - k} L {cx + outer} {cy + k} "
             f"Q {cx + outer} {cy + outer} {cx + k} {cy + outer} L {cx - k} {cy + outer} "
             f"Q {cx - outer} {cy + outer} {cx - outer} {cy + k} Z")
    b = f"M {cx - bore} {cy} a {bore} {bore} 0 1 0 {2 * bore} 0 a {bore} {bore} 0 1 0 {-2 * bore} 0 Z"
    return f'<path d="{o} {b}" fill="{ink}" fill-rule="evenodd"/>'


def table(x0, y0, angle, length, width, top, side, bg, gap=8, serifs=False, cradle=True, thickness=0.42):
    """The couch in local coordinates (x along the table, y across), rotated by `angle` about (x0, y0).
    Two tones: top surface + a thinner side face below it, so it reads as a slab, not a stick.
    serifs=True adds cross bars at both ends (head cradle / foot), so the table reads as a serif 'I'."""
    t = f'transform="translate({x0} {y0}) rotate({angle})"'
    w, s = width, width * thickness
    parts = []
    shapes = [f'<rect x="0" y="{-w / 2}" width="{length}" height="{w + s}" rx="{w / 3}"/>']
    if serifs:
        shapes.append(f'<rect x="-6" y="{-w * 0.95}" width="{w * 0.55}" height="{w * 1.9 + s}" rx="{w * 0.2}"/>')
        shapes.append(f'<rect x="{length - w * 0.55 + 6}" y="{-w * 0.95}" width="{w * 0.55}" height="{w * 1.9 + s}" rx="{w * 0.2}"/>')
    cut = "".join(sh.replace("/>", f' fill="{bg}" stroke="{bg}" stroke-width="{2 * gap}"/>') for sh in shapes)
    parts.append(f"<g {t}>{cut}")
    # side face (darker) then top face
    parts.append(f'<rect x="0" y="{-w / 2}" width="{length}" height="{w + s}" rx="{w / 3}" fill="{side}"/>')
    parts.append(f'<rect x="0" y="{-w / 2}" width="{length}" height="{w}" rx="{w / 3}" fill="{top}"/>')
    if serifs:
        for x in (-6, length - w * 0.55 + 6):
            parts.append(f'<rect x="{x}" y="{-w * 0.95}" width="{w * 0.55}" height="{w * 1.9 + s}" rx="{w * 0.2}" fill="{side}"/>')
            parts.append(f'<rect x="{x}" y="{-w * 0.95}" width="{w * 0.55}" height="{w * 1.9}" rx="{w * 0.2}" fill="{top}"/>')
    if cradle and not serifs:  # head cradle: a shallow cup at the gantry end
        parts.append(f'<path d="M {w * 0.25} {-w * 0.5} q {w * 0.4} {w * 0.55} 0 {w}" fill="none" stroke="{side}" '
                     f'stroke-width="{w * 0.16}" stroke-linecap="round"/>')
    parts.append("</g>")
    return "".join(parts)


def column(x, y_top, y_floor, colour, bg, width=30, gap=6):
    """Vertical column + floor plate under the table (world-vertical, like a real CT table)."""
    col = (f'M {x - width * 0.32} {y_top} L {x + width * 0.32} {y_top} L {x + width * 0.5} {y_floor - 6} '
           f'L {x - width * 0.5} {y_floor - 6} Z')
    plate = f'<rect x="{x - width * 1.1}" y="{y_floor - 7}" width="{width * 2.2}" height="9" rx="4.5" fill="{colour}"/>'
    return (f'<path d="{col}" fill="{bg}" stroke="{bg}" stroke-width="{2 * gap}"/>'
            f'<path d="{col}" fill="{colour}"/>{plate}')


def couch_perspective(far, near, w_far, w_near, top, edge, bg, gap=8, depth=9, headrest=True):
    """A flat CT couch seen in perspective: a board from `far` (inside the bore) to `near` (towards the viewer),
    widening with distance, with a thin darker front/side edge and small corner radii (a board, not a pipe)."""
    (fx, fy), (nx, ny) = far, near
    dx, dy = nx - fx, ny - fy
    L = math.hypot(dx, dy)
    px, py = -dy / L, dx / L  # unit normal
    q = [(fx + px * w_far / 2, fy + py * w_far / 2), (nx + px * w_near / 2, ny + py * w_near / 2),
         (nx - px * w_near / 2, ny - py * w_near / 2), (fx - px * w_far / 2, fy - py * w_far / 2)]
    def poly(pts, **kw):
        attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" {attrs} stroke-linejoin="round"/>'
    low = [(x, y + depth) for x, y in q]
    side = [q[1], q[2], low[2], low[1]]        # front edge (near end)
    under = [q[2], q[3], low[3], low[2]]       # long side edge (visible below)
    out = [poly(q + low[::-1], fill=bg, stroke=bg, stroke_width=2 * gap)]
    out += [poly(under, fill=edge, stroke=edge, stroke_width=3), poly(side, fill=edge, stroke=edge, stroke_width=3),
            poly(q, fill=top, stroke=top, stroke_width=3)]
    if headrest:  # shallow head cradle line near the far end
        hx, hy = fx + dx * 0.10, fy + dy * 0.10
        a, b = (hx + px * w_far * 0.32, hy + py * w_far * 0.32), (hx - px * w_far * 0.32, hy - py * w_far * 0.32)
        out.append(f'<path d="M {a[0]:.1f} {a[1]:.1f} Q {hx + dx / L * 7:.1f} {hy + dy / L * 7:.1f} {b[0]:.1f} {b[1]:.1f}" '
                   f'fill="none" stroke="{edge}" stroke-width="3" stroke-linecap="round"/>')
    return "".join(out)


def brain(ink, cx, cy, s, width=3.0):
    def p(x, y):
        return f"{cx + s * x:.1f} {cy + s * y:.1f}"
    outline = (f"M {p(0, -27)} C {p(13, -28)} {p(22, -18)} {p(22, -3)} C {p(22, 14)} {p(14, 26)} {p(0, 27)} "
               f"C {p(-14, 26)} {p(-22, 14)} {p(-22, -3)} C {p(-22, -18)} {p(-13, -28)} {p(0, -27)} Z")
    inner = (f"M {p(0, -27)} L {p(0, 27)} M {p(-15, -14)} q {s * 5:.1f} {s * 3:.1f} {s * 2:.1f} {s * 9:.1f} "
             f"M {p(15, -14)} q {-s * 5:.1f} {s * 3:.1f} {-s * 2:.1f} {s * 9:.1f} "
             f"M {p(-17, 6)} q {s * 6:.1f} {-s:.1f} {s * 7:.1f} {s * 7:.1f} M {p(17, 6)} q {-s * 6:.1f} {-s:.1f} {-s * 7:.1f} {s * 7:.1f}")
    st = f'fill="none" stroke="{ink}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'
    return f'<path d="{outline}" {st}/><path d="{inner}" {st}/>'


def patient(x0, y0, angle, length, body, bg, gap=5):
    """A lying patient seen from above along the table: head (in the bore), shoulders, body."""
    t = f'transform="translate({x0} {y0}) rotate({angle})"'
    shape = (f'<path d="M 30 -17 Q 44 -21 60 -18 L {length} -14 Q {length + 10} 0 {length} 14 L 60 18 '
             f'Q 44 21 30 17 Q 24 0 30 -17 Z"/>')
    cut = shape.replace("/>", f' fill="{bg}" stroke="{bg}" stroke-width="{2 * gap}"/>')
    return f"<g {t}>{cut}{shape.replace('/>', f' fill=\"{body}\"/>')}</g>"


def variant(name, bg, ink):
    side = ACCENT_DARK
    if name == "F_serif_I":
        # table as a serif I: head cradle and foot as cross bars, long enough to read as a letter
        return (ring(ink, 112, 112, 80, 46, "rounded")
                + table(104, 104, 45, 150, 34, ACCENT, side, bg, serifs=True))
    if name == "G_slab_column":
        # long two-tone couch leaving the bore, standing on a column + floor plate
        return (ring(ink, 108, 104, 80, 46, "rounded")
                + column(196, 188, 244, ink, bg)
                + table(100, 96, 38, 158, 34, ACCENT, side, bg))
    if name == "H_patient":
        # patient lying on the couch, head with brain in the bore
        return (ring(ink, 108, 104, 80, 48, "rounded")
                + table(96, 92, 40, 160, 38, ACCENT, side, bg, cradle=False)
                + patient(96, 92, 40, 150, ink, bg)
                + brain(bg, 100, 96, 0.62, width=2.6))
    if name == "J_round_slab":
        # round gantry (strongest Q) + long two-tone couch
        return (ring(ink, 110, 106, 82, 46, "circle")
                + table(102, 98, 42, 160, 36, ACCENT, side, bg))
    if name == "K_brain_slab":
        # brain in the bore, couch starting below the head, on a column
        return (ring(ink, 108, 100, 80, 50, "rounded")
                + brain(ACCENT, 100, 92, 1.15)
                + column(200, 186, 244, ink, bg)
                + table(110, 118, 36, 150, 30, ink, "#8d96a0" if bg == DARK_BG else "#4a5561", bg))
    if name == "L_flat_couch":
        return (ring(ink, 108, 100, 80, 48, "rounded")
                + column(206, 208, 250, ink, bg, width=26)
                + couch_perspective((100, 96), (214, 196), 26, 46, ACCENT, side, bg))
    if name == "M_flat_brain":
        return (ring(ink, 108, 98, 80, 50, "rounded")
                + brain(ACCENT, 100, 84, 1.05)
                + column(206, 208, 250, ink, bg, width=26)
                + couch_perspective((104, 110), (214, 198), 26, 46, ink, "#8d96a0" if bg == DARK_BG else "#4a5561", bg))
    if name == "N_round_flat":
        return (ring(ink, 108, 100, 82, 48, "circle")
                + column(206, 208, 250, ink, bg, width=26)
                + couch_perspective((100, 96), (214, 196), 26, 46, ACCENT, side, bg))
    raise ValueError(name)


NAMES = ["F_serif_I", "G_slab_column", "K_brain_slab", "L_flat_couch", "M_flat_brain", "N_round_flat"]
LABELS = {"F_serif_I": "F  table as serif I", "G_slab_column": "G  couch on a column",
          "H_patient": "H  patient, head in bore", "J_round_slab": "J  round gantry, long couch",
          "K_brain_slab": "K  brain in bore + couch", "L_flat_couch": "L  flat couch, perspective",
          "M_flat_brain": "M  flat couch + brain", "N_round_flat": "N  round gantry, flat couch"}


def svg_tile(name, bg, ink):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">'
            f'<rect width="{SIZE}" height="{SIZE}" rx="20" fill="{bg}"/>{variant(name, bg, ink)}</svg>')


def sheet():
    gap = 24
    w, h = gap + len(NAMES) * (SIZE + gap), gap + 2 * (SIZE + 44)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             f'<defs><clipPath id="tile"><rect width="{SIZE}" height="{SIZE}" rx="20"/></clipPath></defs>',
             f'<rect width="{w}" height="{h}" fill="#e9edf1"/>']
    for row, (bg, ink) in enumerate(((DARK_BG, INK_DARK), (LIGHT_BG, INK_LIGHT))):
        for i, n in enumerate(NAMES):
            x, y = gap + i * (SIZE + gap), gap + row * (SIZE + 44)
            parts.append(f'<g transform="translate({x} {y})"><rect width="{SIZE}" height="{SIZE}" rx="20" fill="{bg}"/>'
                         f'<g clip-path="url(#tile)">{variant(n, bg, ink)}</g></g>')
            parts.append(f'<text x="{x + 6}" y="{y + SIZE + 24}" font-family="Arial" font-size="17" '
                         f'fill="#1f2933">{LABELS[n]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    (OUT / "logo2_sheet.svg").write_text(sheet(), encoding="utf-8")
    for n in NAMES:
        for tag, bg, ink in (("dark", DARK_BG, INK_DARK), ("light", LIGHT_BG, INK_LIGHT)):
            (OUT / f"logo2_{n}_{tag}.svg").write_text(svg_tile(n, bg, ink), encoding="utf-8")
    print("written to", OUT)
