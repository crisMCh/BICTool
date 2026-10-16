"""BICTool logo, round 3 (the user's chosen direction): a CT gantry forming a Q, a brain in the bore and the
patient couch in perspective leaving the bore as the tail of the Q. Clean vector redraw, no text.

Writes logo3_<theme>.svg (with background) and logo3_<theme>_transparent.svg, themes:
  light  navy / teal on white (the reference colours)
  dark   BICTool app palette on the app background
Run:  .venv/Scripts/python.exe design/logo/make_logo3.py
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
THEMES = {
    "light": dict(bg="#ffffff", body="#0e2f5c", brain="#7cc5d8", top="#17899b", side="#1c6b7d", line="#ffffff"),
    "dark": dict(bg="#11151a", body="#e7eaed", brain="#49c8de", top="#49c8de", side="#2a8fa3", line="#11151a"),
}
V = 512                                  # viewBox size
CX, CY = 256, 214                        # gantry centre
BX, BY = 256, 206                        # bore centre (slightly above the gantry centre)
HOLE_R, RING_OUT, RING_IN = 96, 88, 72   # body hole, inner navy ring


def superellipse(cx, cy, a, b, n=2.7, steps=180):
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s) ** (2 / n), s)
        pts.append(f"{cx + x:.1f} {cy + y:.1f}")
    return "M " + " L ".join(pts) + " Z"


def circle_path(cx, cy, r):
    return f"M {cx - r} {cy} a {r} {r} 0 1 0 {2 * r} 0 a {r} {r} 0 1 0 {-2 * r} 0 Z"


def gantry(t):
    body = f'<path d="{superellipse(CX, CY, 176, 158, n=2.3)} {circle_path(BX, BY, HOLE_R)}" fill="{t["body"]}" fill-rule="evenodd"/>'
    ring = f'<path d="{circle_path(BX, BY, RING_OUT)} {circle_path(BX, BY, RING_IN)}" fill="{t["body"]}" fill-rule="evenodd"/>'
    return body + ring


def brain(t, cx=BX, cy=BY - 10, s=1.18):
    """Coronal-ish brain: two lobed hemispheres (unions of circles) with a gap for the midline, flat base."""
    lobes = [(-30, 6, 15), (-28, -12, 15), (-14, -24, 15), (-14, 14, 13), (-16, -4, 16),
             (-36, -2, 11), (-26, 20, 9)]
    circles = []
    for x, y, r in lobes:
        for sign in (-1, 1):
            if sign == -1:
                circles.append((cx + s * x, cy + s * y, s * r))
            else:
                circles.append((cx - s * x, cy + s * y, s * r))
    clip_id = f"brainclip_{t['bg'][1:]}"
    clip = (f'<clipPath id="{clip_id}"><rect x="{cx - 60 * s}" y="{cy - 60 * s}" width="{120 * s}" '
            f'height="{88 * s}"/></clipPath>')
    blobs = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>' for x, y, r in circles)
    gap = (f'<rect x="{cx - 2.6 * s}" y="{cy - 44 * s}" width="{5.2 * s}" height="{80 * s}" fill="{t["bg"]}"/>')
    return (f'<defs>{clip}</defs><g clip-path="url(#{clip_id})"><g fill="{t["brain"]}">{blobs}</g>{gap}</g>')


def couch(t, gap=11):
    """Flat couch in perspective: narrow far end in the bore, wide near end outside the gantry, darker side and
    front faces (thickness grows towards the viewer), a thin highlight line along the near edges."""
    FL, FR = (210, 252), (270, 247)       # far end (in the bore)
    NL, NR = (322, 384), (436, 362)       # near end (towards the viewer)
    d_far, d_near = 4, 30                 # thickness at the far and near end
    NLb, NRb = (NL[0], NL[1] + d_near), (NR[0], NR[1] + d_near)
    FLb = (FL[0], FL[1] + d_far)

    def poly(pts):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    outline = [FL, FR, NR, NRb, NLb, FLb]
    top = [FL, FR, NR, NL]
    side = [FL, NL, NLb, FLb]             # long left side face
    front = [NL, NR, NRb, NLb]            # near end face

    def inset(p, q, k):                   # point k of the way from p to q
        return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)

    # highlight: along the left edge and the front edge, slightly inside the top face
    c = (sum(x for x, _ in top) / 4, sum(y for _, y in top) / 4)
    shrink = [inset(p, c, 0.06) for p in top]
    hl = f"M {shrink[0][0]:.1f} {shrink[0][1]:.1f} L {shrink[3][0]:.1f} {shrink[3][1]:.1f} L {shrink[2][0]:.1f} {shrink[2][1]:.1f}"
    j = 'stroke-linejoin="round"'
    return (f'<polygon points="{poly(outline)}" fill="{t["bg"]}" stroke="{t["bg"]}" stroke-width="{2 * gap}" {j}/>'
            f'<polygon points="{poly(side)}" fill="{t["side"]}" stroke="{t["side"]}" stroke-width="6" {j}/>'
            f'<polygon points="{poly(front)}" fill="{t["side"]}" stroke="{t["side"]}" stroke-width="6" {j}/>'
            f'<polygon points="{poly(top)}" fill="{t["top"]}" stroke="{t["top"]}" stroke-width="6" {j}/>'
            f'<path d="{hl}" fill="none" stroke="{t["line"]}" stroke-width="3.5" stroke-linecap="round" {j}/>')


def logo(theme, background=True):
    t = THEMES[theme]
    bg = f'<rect width="{V}" height="{V}" fill="{t["bg"]}"/>' if background else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{V}" height="{V}" viewBox="0 0 {V} {V}">'
            f'{bg}<g transform="translate(-6 18)">{gantry(t)}{brain(t)}{couch(t)}</g></svg>')


if __name__ == "__main__":
    for theme in THEMES:
        (OUT / f"logo3_{theme}.svg").write_text(logo(theme), encoding="utf-8")
        (OUT / f"logo3_{theme}_transparent.svg").write_text(logo(theme, background=False), encoding="utf-8")
    print("written to", OUT)
