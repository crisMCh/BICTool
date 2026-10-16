"""BICTool logo, round 4: the user's reference (GPT image 3) redrawn with consistent perspective.

Geometry (coordinates of the 1256 px reference, cropped by the viewBox):
- One vanishing point VP (upper left). The couch's near and far edges are horizontal, its long edges and the
  bottom edges run to VP, and the thickness shrinks towards VP. That is what makes the couch look natural.
- The gantry is a short cylinder whose axis also points to VP: the back outer rim shows at the upper left, and
  inside the bore the tunnel wall shows at the lower right. The couch therefore runs along the tunnel axis.
- Two colours like the reference (gantry + couch, brain), white gaps between parts; a light strip on the gantry.

Run:  .venv/Scripts/python.exe design/logo/make_logo4.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent
THEMES = {
    "light": dict(bg="#ffffff", main="#243d55", back="#162839", tunnel="#d9e1e8", brain="#4a9ea8",
                  sulci="#ffffff", strip="#ffffff"),
    "dark": dict(bg="#11151a", main="#e7eaed", back="#9aa4ae", tunnel="#2b323a", brain="#49c8de",
                 sulci="#11151a", strip="#49c8de"),
}
VP = (0.0, 80.0)                 # vanishing point (upper left, outside the picture)
C, R_OUT, R_BORE = (588.0, 462.0), 283.0, 200.0
DEPTH = 0.07                     # gantry depth: back face = front face moved 10 % towards VP
VIEWBOX = "240 140 710 710"


def towards_vp(p, k):
    return (p[0] + k * (VP[0] - p[0]), p[1] + k * (VP[1] - p[1]))


def circle(c, r):
    return f"M {c[0] - r:.1f} {c[1]:.1f} a {r:.1f} {r:.1f} 0 1 0 {2 * r:.1f} 0 a {r:.1f} {r:.1f} 0 1 0 {-2 * r:.1f} 0 Z"


def gantry_flat(t, inner_ring):
    """Flat gantry (no depth): ring, optional thin bore-cover ring with a white gap, light strip on top."""
    parts = [f'<path d="{circle(C, R_OUT)} {circle(C, R_BORE)}" fill="{t["main"]}" fill-rule="evenodd"/>']
    if inner_ring:
        parts.append(f'<path d="{circle(C, R_BORE - 16)} {circle(C, R_BORE - 34)}" fill="{t["main"]}" fill-rule="evenodd"/>')
    parts.append(f'<rect x="{C[0] - 42}" y="{C[1] - R_OUT + 30}" width="84" height="16" rx="8" fill="{t["strip"]}"/>')
    return "".join(parts)


def gantry(t):
    back_c = towards_vp(C, DEPTH)
    back_r_out, back_r_bore = R_OUT * (1 - DEPTH), R_BORE * (1 - DEPTH)
    parts = [
        # back rim of the cylinder (only the upper-left crescent stays visible)
        f'<path d="{circle(back_c, back_r_out)} {circle(back_c, back_r_bore)}" fill="{t["back"]}" fill-rule="evenodd"/>',
        # front face: ring
        f'<path d="{circle(C, R_OUT)} {circle(C, R_BORE)}" fill="{t["main"]}" fill-rule="evenodd"/>',
        # tunnel wall: inside the front bore, outside the back opening (lower-right crescent)
        f'<path d="{circle(C, R_BORE)} {circle(back_c, back_r_bore)}" fill="{t["tunnel"]}" fill-rule="evenodd"/>',
        # light strip on top of the gantry
        f'<rect x="{C[0] - 42}" y="{C[1] - R_OUT + 30}" width="84" height="16" rx="8" fill="{t["strip"]}"/>',
    ]
    return "".join(parts)


def brain(t, cx=588.0, cy=420.0, s=1.0):
    """Front view of a brain: two lobed hemispheres (circles), midline gap, white sulci, flat-ish base."""
    lobes = [(-22, -62, 24), (-50, -44, 26), (-70, -10, 26), (-62, 26, 26), (-36, 54, 26), (-12, 64, 18),
             (-30, -10, 34), (-16, 26, 30)]
    blobs = []
    for x, y, r in lobes:
        blobs.append(f'<circle cx="{cx + s * x:.1f}" cy="{cy + s * y:.1f}" r="{s * r:.1f}"/>')
        blobs.append(f'<circle cx="{cx - s * x:.1f}" cy="{cy + s * y:.1f}" r="{s * r:.1f}"/>')
    gap = f'<rect x="{cx - 5 * s:.1f}" y="{cy - 84 * s:.1f}" width="{10 * s:.1f}" height="{166 * s:.1f}" fill="{t["bg"]}"/>'
    sulci = []
    for sign in (-1, 1):
        def px(x):
            return cx + sign * s * x
        sulci.append(f'M {px(30):.1f} {cy - 36 * s:.1f} q {sign * s * 14:.1f} {s * 10:.1f} {sign * s * 4:.1f} {s * 26:.1f}')
        sulci.append(f'M {px(30):.1f} {cy + 30 * s:.1f} q {sign * s * 16:.1f} {-s * 14:.1f} {sign * s * 34:.1f} {s * 2:.1f}')
    sul = (f'<path d="{" ".join(sulci)}" fill="none" stroke="{t["sulci"]}" stroke-width="{7 * s:.1f}" '
           f'stroke-linecap="round"/>')
    return f'<g fill="{t["brain"]}">{"".join(blobs)}</g>{gap}{sul}'


def couch(t, gap=16):
    """Couch board in one-point perspective towards VP: horizontal near/far edges, long edges to VP."""
    y_near, y_far = 716.0, 521.0
    NL, NR = (712.0, y_near), (906.0, y_near)
    k = (y_near - y_far) / (y_near - VP[1])                  # where the far edge lies on the lines to VP
    FL, FR = towards_vp(NL, k), towards_vp(NR, k)
    thick = 62.0
    NLb, NRb = (NL[0], NL[1] + thick), (NR[0], NR[1] + thick)
    # the side face tapers more than strict perspective would give (as in the reference): thin far end
    FLb = (FL[0], FL[1] + 16.0)

    def poly(pts):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    j = 'stroke-linejoin="round"'
    outline = [FL, FR, NR, NRb, NLb, FLb]
    top = [FL, FR, NR, NL]
    side = [FL, NL, NLb, FLb]
    front = [NL, NR, NRb, NLb]
    sep = 7  # white separation lines between the faces (like the reference)
    return (f'<polygon points="{poly(outline)}" fill="{t["bg"]}" stroke="{t["bg"]}" stroke-width="{2 * gap}" {j}/>'
            f'<polygon points="{poly(side)}" fill="{t["main"]}" stroke="{t["main"]}" stroke-width="10" {j}/>'
            f'<polygon points="{poly(front)}" fill="{t["main"]}" stroke="{t["main"]}" stroke-width="10" {j}/>'
            f'<polygon points="{poly(top)}" fill="{t["main"]}" stroke="{t["bg"]}" stroke-width="{sep}" {j}/>')


def logo(theme, background=True, style="flat"):
    """style: "flat" (ring + light strip), "flat_ring" (plus a thin bore ring), "depth" (round 4, 3D gantry)."""
    t = THEMES[theme]
    x, y, w, h = VIEWBOX.split()
    bg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{t["bg"]}"/>' if background else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="{VIEWBOX}">'
            f'{bg}{gantry(t) if style == "depth" else gantry_flat(t, style == "flat_ring")}{brain(t, s=0.9 if style == "flat_ring" else 1.0)}{couch(t)}</svg>')


if __name__ == "__main__":
    for theme in THEMES:
        (OUT / f"logo4_{theme}.svg").write_text(logo(theme, style="depth"), encoding="utf-8")
        for style in ("flat", "flat_ring"):
            (OUT / f"logo5_{style}_{theme}.svg").write_text(logo(theme, style=style), encoding="utf-8")
            (OUT / f"logo5_{style}_{theme}_transparent.svg").write_text(logo(theme, False, style), encoding="utf-8")
    print("written to", OUT)
