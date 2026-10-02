"""
ui_kit.py - light studio UI primitives for the Humanizer desktop app.

Design system for a soft, paper-white desktop app:
- Light page canvas with clean soft blue-grey page background
- Cards and buttons carry smooth anti-aliased geometry and soft diffused shadows
- High-fidelity supersampled rendering via PIL for pixel-perfect curves
- 1px vector glyphs, spring-eased motion, and native ClearType text
"""
import math
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageTk, ImageColor


# ----------------------------------------------------------------- palette
INK         = "#f1f4fb"   # page background
TRAY        = "#ffffff"   # card body
CORE        = "#ffffff"   # card core
EDITOR      = "#ffffff"   # text surface
HAIRLINE    = "#e2e8f0"   # 1px border
HAIRLINE_2  = "#cbd5e1"
TEXT        = "#0f172a"   # headings / dark slate
MUTED       = "#64748b"   # secondary text
FAINT       = "#94a3b8"   # tertiary / stats
ACCENT      = "#2563eb"   # primary blue
ACCENT_2    = "#1d61f2"
ACCENT_3    = "#3b82f6"
ACCENT_TOP  = "#2b7af1"   # gradient top for blue surfaces
ACCENT_BOT  = "#1d5ed8"   # gradient bottom for blue surfaces
ACCENT_SOFT = "#e8f0fe"
GOOD        = "#10b981"   # emerald green for detectors
GOOD_BG     = "#ecfdf5"   # emerald light tint
GOOD_BORDER = "#a7f3d0"
WARN        = "#f59e0b"
BAD         = "#ef4444"
BTN         = ACCENT
GHOST       = "#ffffff"   # soft button fill
GHOST_HI    = "#f8fafc"   # soft button hover
GHOST_RING  = "#e2e8f0"
TRACK       = "#f1f5f9"   # progress / segmented track
SHADOW      = "#64748b"


# ------------------------------------------------------------- color utils
def _rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    """Blend hex c1 -> c2 by t (0..1)."""
    r1, g1, b1 = _rgb(c1)
    r2, g2, b2 = _rgb(c2)
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % (round(r1 + (r2 - r1) * t),
                              round(g1 + (g2 - g1) * t),
                              round(b1 + (b2 - b1) * t))


def alpha_on(over, bg, a):
    """Fake alpha: blend `over` onto `bg` at strength a."""
    return mix(bg, over, a)


# ---------------------------------------------------------- motion curves
def ease_out_cubic(t):
    return 1.0 - pow(1.0 - t, 3)


def ease_out_back(t, s=1.15):
    c3 = s + 1.0
    return 1 + c3 * pow(t - 1, 3) + s * pow(t - 1, 2)


def rounded_points(x0, y0, x1, y1, r, steps=9):
    """Polygon points for a rounded rectangle (fallback)."""
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    pts = []

    def arc(cx, cy, a0, a1):
        for i in range(steps + 1):
            a = a0 + (a1 - a0) * i / steps
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    arc(x0 + r, y0 + r, math.pi, math.pi * 1.5)
    arc(x1 - r, y0 + r, math.pi * 1.5, math.pi * 2.0)
    arc(x1 - r, y1 - r, 0.0, math.pi * 0.5)
    arc(x0 + r, y1 - r, math.pi * 0.5, math.pi)
    return pts


# ---------------------------------------------------- PIL high-res rendering
_IMAGE_CACHE = {}


def get_rounded_rect_img(w, h, r, fill, outline=None, outline_width=1,
                         parent_bg="#ffffff", shadow=0, shadow_dy=1.5,
                         shadow_color=None, master=None):
    """
    Renders an anti-aliased rounded rectangle with supersampling and soft shadow.
    Returns cached ImageTk.PhotoImage.
    """
    if master is None:
        master = tk._default_root
    w, h, r = max(2, int(w)), max(2, int(h)), max(0, int(r))
    key = (id(master), w, h, r, str(fill), str(outline), outline_width, str(parent_bg),
           shadow, shadow_dy, str(shadow_color))
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key][0]

    scale = 3
    sw, sh = w * scale, h * scale
    sr = min(r * scale, min(sw, sh) // 2)

    bg_color = parent_bg or "#ffffff"
    img = Image.new("RGBA", (sw, sh), bg_color)
    draw = ImageDraw.Draw(img)

    # Soft shadow
    if shadow > 0:
        smask = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(smask)
        pad = int(1.5 * scale)
        s_col = shadow_color or (100, 116, 139, 35)
        sdraw.rounded_rectangle(
            [pad, pad + int(shadow_dy * scale),
             sw - pad, sh - pad + int(shadow_dy * scale)],
            radius=sr, fill=s_col
        )
        smask = smask.filter(ImageFilter.GaussianBlur(shadow * scale * 0.7))
        img = Image.alpha_composite(img, smask)
        draw = ImageDraw.Draw(img)

    inset = 1 * scale
    rect_box = [inset, inset, sw - inset, sh - inset]

    if isinstance(fill, (list, tuple)) and len(fill) == 2:
        top_c, bot_c = fill
        grad = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(grad)
        c1 = ImageColor.getrgb(top_c)
        c2 = ImageColor.getrgb(bot_c)
        for y in range(sh):
            t = y / max(1, sh - 1)
            rc = int(c1[0] + (c2[0] - c1[0]) * t)
            gc = int(c1[1] + (c2[1] - c1[1]) * t)
            bc = int(c1[2] + (c2[2] - c1[2]) * t)
            gdraw.line([(0, y), (sw, y)], fill=(rc, gc, bc, 255))
        mask = Image.new("L", (sw, sh), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle(rect_box, radius=sr, fill=255)
        img.paste(grad, (0, 0), mask)
        draw = ImageDraw.Draw(img)
        if outline:
            draw.rounded_rectangle(rect_box, radius=sr, outline=outline,
                                   width=max(1, int(outline_width * scale)))
    else:
        draw.rounded_rectangle(
            rect_box, radius=sr, fill=fill,
            outline=outline if outline else None,
            width=max(1, int(outline_width * scale)) if outline else 0
        )

    final_img = img.resize((w, h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(final_img, master=master)
    _IMAGE_CACHE[key] = (photo, final_img)
    return photo


def get_logo_img(w=32, h=30, parent_bg="#ffffff", master=None):
    """Two slanted parallel rounded leaves with two-tone sky/royal blue split matching the brand mark."""
    if master is None:
        master = tk._default_root
    key = ("logo", id(master), w, h, parent_bg)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key][0]
    scale = 4
    sw, sh = w * scale, h * scale
    img = Image.new("RGBA", (sw, sh), parent_bg)

    lw, lh = int(9 * scale), int(24 * scale)
    lr = lw // 2

    for x_off, y_off in [(int(6 * scale), int(3 * scale)), (int(16 * scale), int(1 * scale))]:
        leaf = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(leaf)
        ldraw.rounded_rectangle([0, 0, lw - 1, lh - 1], radius=lr, fill="#1d4ed8")

        top_mask = Image.new("L", (lw, lh), 0)
        mdraw = ImageDraw.Draw(top_mask)
        mdraw.rounded_rectangle([0, 0, lw - 1, int(lh * 0.58)], radius=lr, fill=255)
        curve_pts = [(0, int(lh * 0.42)), (lw, int(lh * 0.58)), (0, int(lh * 0.58))]
        mdraw.polygon(curve_pts, fill=255)

        sky = Image.new("RGBA", (lw, lh), "#38bdf8")
        leaf.paste(sky, (0, 0), top_mask)

        rot = leaf.rotate(18, resample=Image.Resampling.BICUBIC, expand=True)
        img.paste(rot, (x_off, y_off), rot)

    final_img = img.resize((w, h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(final_img, master=master)
    _IMAGE_CACHE[key] = (photo, final_img)
    return photo


def get_script_signoff_img(w=180, h=36, bg="#f1f4fb", master=None):
    """Render 'Write like a human' in Segoe Script with an elegant underline swoosh."""
    if master is None:
        master = tk._default_root
    key = ("signoff", id(master), w, h, bg)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key][0]
    scale = 3
    sw, sh = w * scale, h * scale
    img = Image.new("RGBA", (sw, sh), bg)

    font = None
    import os
    from PIL import ImageFont
    for fpath in (r"C:\Windows\Fonts\segoesc.ttf", r"C:\Windows\Fonts\Inkfree.ttf",
                  r"C:\Windows\Fonts\BRADHITC.TTF", r"C:\Windows\Fonts\comic.ttf"):
        if os.path.exists(fpath):
            try:
                font = ImageFont.truetype(fpath, int(13.5 * scale))
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    color = (139, 152, 173, 255)
    text_img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(text_img)
    tdraw.text((int(14 * scale), int(4 * scale)), "Write like a human", font=font, fill=color)

    sw_pts = [
        (int(24 * scale), int(26 * scale)),
        (int(90 * scale), int(24 * scale)),
        (int(165 * scale), int(21 * scale))
    ]
    tdraw.line(sw_pts, fill=color, width=int(1.3 * scale), joint="round")

    rot = text_img.rotate(2.5, resample=Image.Resampling.BICUBIC, center=(int(90 * scale), int(18 * scale)))
    img = Image.alpha_composite(img, rot)

    final_img = img.resize((w, h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(final_img, master=master)
    _IMAGE_CACHE[key] = (photo, final_img)
    return photo


class ScriptSignoff(tk.Canvas):
    """Handwritten signoff widget with underline swoosh."""

    def __init__(self, master, bg=None, **kw):
        self._bg = bg or INK
        super().__init__(master, width=180, height=36, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self.bind("<Configure>", lambda e: self._paint())
        self.after_idle(self._paint)

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 10:
            return
        self.delete("all")
        photo = get_script_signoff_img(w, h, bg=self._bg, master=self)
        self.create_image(0, 0, image=photo, anchor="nw")
        self._photo = photo


def tracked(text, gap="\u2009"):
    """Emulate letter-spacing with thin spaces."""
    return gap.join(list(text))


def tracked_wide(text, gap=" "):
    """Wider tracking for eyebrow labels."""
    return gap.join(list(text))


# ------------------------------------------------------------------ fonts
class Fonts:
    display = "Segoe UI"
    text = "Segoe UI"
    script = "Segoe Script"

    @classmethod
    def resolve(cls):
        try:
            fams = set(tkfont.families())
        except Exception:
            fams = set()
        for cand in ("Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI"):
            if cand in fams:
                cls.display = cand
                break
        for cand in ("Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI"):
            if cand in fams:
                cls.text = cand
                break
        for cand in ("Segoe Script", "Ink Free", "Bradley Hand ITC", "Comic Sans MS"):
            if cand in fams:
                cls.script = cand
                break

    @classmethod
    def get(cls, kind="text"):
        return getattr(cls, kind, cls.text)

    @classmethod
    def measure(cls, size, weight="normal", kind="text"):
        key = (kind, size, weight)
        w = cls._cache.get(key)
        if w is None:
            try:
                w = tkfont.Font(family=cls.get(kind), size=size,
                                weight=weight).measure
            except Exception:
                w = lambda _t: len(_t) * size * 0.55
            cls._cache[key] = w
        return w

    _cache = {}


def label(parent, text="", size=10, color=TEXT, weight="normal",
          family="text", bg=None, anchor="w", **kw):
    if bg is None:
        try:
            bg = parent.cget("bg")
        except Exception:
            bg = INK
    return tk.Label(parent, text=text, font=(Fonts.get(family), size, weight),
                    bg=bg, fg=color, anchor=anchor, **kw)


def divider(parent, color=HAIRLINE, pady=0):
    f = tk.Frame(parent, bg=color, height=1)
    f.pack(fill="x", pady=pady)
    return f


# --------------------------------------------------------- vector icons
def get_icon_photo(kind, s, color, master=None, width=1.4):
    """Render vector icons using 4x supersampled PIL with smooth anti-aliased edges."""
    if master is None:
        master = tk._default_root
    key = ("icon", id(master), kind, int(s), str(color), float(width))
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key][0]

    scale = 4
    sw = max(8, int(s * scale))
    sh = max(8, int(s * scale))
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    w = max(1, int(width * scale))
    k = sw / 2.0
    cx, cy = k, k

    col_str = color or "#64748b"
    try:
        rgba = ImageColor.getrgb(col_str)
        if len(rgba) == 3:
            rgba = rgba + (255,)
    except Exception:
        rgba = (100, 116, 139, 255)

    def line(pts, line_w=None, width=None):
        lw = width if width is not None else (line_w if line_w is not None else w)
        draw.line(pts, fill=rgba, width=lw, joint="round")

    if kind == "arrow":
        line([(cx - k * 0.7, cy), (cx + k * 0.7, cy)])
        line([(cx + k * 0.2, cy - k * 0.5), (cx + k * 0.7, cy), (cx + k * 0.2, cy + k * 0.5)])
    elif kind == "arrow_left":
        line([(cx + k * 0.7, cy), (cx - k * 0.7, cy)])
        line([(cx - k * 0.2, cy - k * 0.5), (cx - k * 0.7, cy), (cx - k * 0.2, cy + k * 0.5)])
    elif kind == "swap":
        y1 = cy - k * 0.35
        y2 = cy + k * 0.35
        line([(cx - k * 0.75, y1), (cx + k * 0.75, y1)])
        line([(cx + k * 0.35, y1 - k * 0.35), (cx + k * 0.75, y1), (cx + k * 0.35, y1 + k * 0.35)])
        line([(cx + k * 0.75, y2), (cx - k * 0.75, y2)])
        line([(cx - k * 0.35, y2 - k * 0.35), (cx - k * 0.75, y2), (cx - k * 0.35, y2 + k * 0.35)])
    elif kind == "refresh":
        draw.arc([cx - k * 0.8, cy - k * 0.8, cx + k * 0.8, cy + k * 0.8], start=45, end=330, fill=rgba, width=w)
        line([(cx + k * 0.2, cy - k * 0.95), (cx + k * 0.8, cy - k * 0.8), (cx + k * 0.5, cy - k * 0.35)])
    elif kind == "undo":
        line([(cx - k * 0.35, cy - k * 0.7), (cx - k * 0.8, cy - k * 0.35), (cx - k * 0.35, cy)])
        line([(cx - k * 0.75, cy - k * 0.35), (cx + k * 0.1, cy - k * 0.35)])
        draw.arc([cx - k * 0.45, cy - k * 0.35, cx + k * 0.65, cy + k * 0.75], start=270, end=90, fill=rgba, width=w)
    elif kind == "redo":
        line([(cx + k * 0.35, cy - k * 0.7), (cx + k * 0.8, cy - k * 0.35), (cx + k * 0.35, cy)])
        line([(cx + k * 0.75, cy - k * 0.35), (cx - k * 0.1, cy - k * 0.35)])
        draw.arc([cx - k * 0.65, cy - k * 0.35, cx + k * 0.45, cy + k * 0.75], start=90, end=270, fill=rgba, width=w)
    elif kind == "copy":
        draw.rounded_rectangle([cx - k * 0.85, cy - k * 0.85, cx + k * 0.3, cy + k * 0.3], radius=int(k * 0.25), outline=rgba, width=w)
        draw.rounded_rectangle([cx - k * 0.3, cy - k * 0.3, cx + k * 0.85, cy + k * 0.85], radius=int(k * 0.25), fill=(255, 255, 255, 255), outline=rgba, width=w)
    elif kind == "save":
        draw.rounded_rectangle([cx - k * 0.8, cy - k * 0.8, cx + k * 0.8, cy + k * 0.8], radius=int(k * 0.2), outline=rgba, width=w)
        draw.rectangle([cx - k * 0.45, cy - k * 0.8, cx + k * 0.45, cy - k * 0.25], outline=rgba, width=w)
        draw.rectangle([cx - k * 0.5, cy + k * 0.1, cx + k * 0.5, cy + k * 0.8], outline=rgba, width=w)
    elif kind == "open":
        line([(cx - k * 0.85, cy - k * 0.4), (cx - k * 0.3, cy - k * 0.4), (cx - k * 0.1, cy - k * 0.1), (cx + k * 0.85, cy - k * 0.1)])
        line([(cx - k * 0.85, cy - k * 0.4), (cx - k * 0.85, cy + k * 0.75), (cx + k * 0.85, cy + k * 0.75), (cx + k * 0.85, cy - k * 0.1)])
        line([(cx - k * 0.85, cy + k * 0.1), (cx + k * 0.85, cy + k * 0.1)])
    elif kind == "trash":
        line([(cx - k * 0.3, cy - k * 0.85), (cx + k * 0.3, cy - k * 0.85)])
        line([(cx - k * 0.8, cy - k * 0.55), (cx + k * 0.8, cy - k * 0.55)])
        draw.line([(cx - k * 0.65, cy - k * 0.55), (cx - k * 0.5, cy + k * 0.8), (cx + k * 0.5, cy + k * 0.8), (cx + k * 0.65, cy - k * 0.55)], fill=rgba, width=w, joint="round")
        line([(cx - k * 0.2, cy - k * 0.3), (cx - k * 0.16, cy + k * 0.6)])
        line([(cx + k * 0.2, cy - k * 0.3), (cx + k * 0.16, cy + k * 0.6)])
    elif kind == "close":
        line([(cx - k * 0.6, cy - k * 0.6), (cx + k * 0.6, cy + k * 0.6)])
        line([(cx + k * 0.6, cy - k * 0.6), (cx - k * 0.6, cy + k * 0.6)])
    elif kind == "sparkle":
        def star4(stx, sty, sr, ir=0.38):
            pts = []
            for i in range(8):
                ang = i * (math.pi / 4.0) - math.pi / 2.0
                rad = sr if (i % 2 == 0) else (sr * ir)
                pts.append((stx + math.cos(ang) * rad, sty + math.sin(ang) * rad))
            draw.polygon(pts, fill=rgba)
        star4(cx + k * 0.22, cy - k * 0.15, k * 0.65, ir=0.38)
        star4(cx - k * 0.35, cy + k * 0.35, k * 0.50, ir=0.38)
        star4(cx - k * 0.45, cy - k * 0.45, k * 0.28, ir=0.38)
        star4(cx + k * 0.45, cy + k * 0.55, k * 0.22, ir=0.38)
    elif kind == "file":
        draw.line([(cx - k * 0.65, cy - k * 0.9), (cx + k * 0.2, cy - k * 0.9), (cx + k * 0.65, cy - k * 0.45),
                   (cx + k * 0.65, cy + k * 0.9), (cx - k * 0.65, cy + k * 0.9), (cx - k * 0.65, cy - k * 0.9)], fill=rgba, width=w, joint="round")
        draw.line([(cx + k * 0.2, cy - k * 0.9), (cx + k * 0.2, cy - k * 0.45), (cx + k * 0.65, cy - k * 0.45)], fill=rgba, width=w)
        line([(cx - k * 0.35, cy), (cx + k * 0.35, cy)])
        line([(cx - k * 0.35, cy + k * 0.35), (cx + k * 0.35, cy + k * 0.35)])
        line([(cx - k * 0.35, cy + k * 0.65), (cx + k * 0.1, cy + k * 0.65)])
    elif kind == "expand":
        d1, d2 = k * 0.25, k * 0.85
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            line([(cx + sx * d1, cy + sy * d2), (cx + sx * d2, cy + sy * d2), (cx + sx * d2, cy + sy * d1)])
    elif kind == "collapse":
        d1, d2 = k * 0.85, k * 0.25
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            line([(cx + sx * d1, cy + sy * d2), (cx + sx * d2, cy + sy * d2), (cx + sx * d2, cy + sy * d1)])
    elif kind == "sliders":
        for i, yy in enumerate((-k * 0.55, 0, k * 0.55)):
            line([(cx - k * 0.9, cy + yy), (cx + k * 0.9, cy + yy)])
            xx = (-k * 0.25, k * 0.4, -k * 0.4)[i]
            draw.ellipse([cx + xx - k * 0.24, cy + yy - k * 0.24, cx + xx + k * 0.24, cy + yy + k * 0.24], fill=(248, 250, 252, 255), outline=rgba, width=w)
    elif kind == "gear":
        pts = []
        n = 12
        for i in range(n * 2):
            ang = i * (math.pi / n)
            rad = k * (0.88 if i % 2 == 0 else 0.74)
            pts.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))
        draw.polygon(pts, outline=rgba, fill=(255, 255, 255, 255), width=w)
        lpts = [(cx - k * 0.15, cy - k * 0.38), (cx - k * 0.3, cy - k * 0.2), (cx - k * 0.3, cy - k * 0.08),
                (cx - k * 0.45, cy), (cx - k * 0.3, cy + k * 0.08), (cx - k * 0.3, cy + k * 0.2), (cx - k * 0.15, cy + k * 0.38)]
        draw.line(lpts, fill=rgba, width=w, joint="round")
        rpts = [(cx + k * 0.15, cy - k * 0.38), (cx + k * 0.3, cy - k * 0.2), (cx + k * 0.3, cy - k * 0.08),
                (cx + k * 0.45, cy), (cx + k * 0.3, cy + k * 0.08), (cx + k * 0.3, cy + k * 0.2), (cx + k * 0.15, cy + k * 0.38)]
        draw.line(rpts, fill=rgba, width=w, joint="round")
    elif kind == "info":
        disc_c = (142, 169, 202, 255)
        draw.ellipse([cx - k * 0.85, cy - k * 0.85, cx + k * 0.85, cy + k * 0.85], fill=disc_c)
        dot_r = k * 0.12
        draw.ellipse([cx - dot_r, cy - k * 0.5 - dot_r, cx + dot_r, cy - k * 0.5 + dot_r], fill=(255, 255, 255, 255))
        draw.line([(cx, cy - k * 0.15), (cx, cy + k * 0.45)], fill=(255, 255, 255, 255), width=int(1.8 * scale), joint="round")
    elif kind == "thumb_up":
        line([(cx - k * 0.7, cy), (cx - k * 0.7, cy + k * 0.75)], width=w)
        pts = [(cx - k * 0.45, cy + k * 0.75), (cx + k * 0.45, cy + k * 0.75), (cx + k * 0.65, cy + k * 0.55),
               (cx + k * 0.65, cy + k * 0.1), (cx + k * 0.45, cy), (cx + k * 0.1, cy), (cx + k * 0.1, cy - k * 0.3),
               (cx + k * 0.18, cy - k * 0.7), (cx + k * 0.0, cy - k * 0.8), (cx - k * 0.2, cy - k * 0.4),
               (cx - k * 0.2, cy), (cx - k * 0.45, cy), (cx - k * 0.45, cy + k * 0.75)]
        draw.line(pts, fill=rgba, width=w, joint="round")
    elif kind == "thumb_down":
        line([(cx - k * 0.7, cy), (cx - k * 0.7, cy - k * 0.75)], width=w)
        pts = [(cx - k * 0.45, cy - k * 0.75), (cx + k * 0.45, cy - k * 0.75), (cx + k * 0.65, cy - k * 0.55),
               (cx + k * 0.65, cy - k * 0.1), (cx + k * 0.45, cy), (cx + k * 0.1, cy), (cx + k * 0.1, cy + k * 0.3),
               (cx + k * 0.18, cy + k * 0.7), (cx + k * 0.0, cy + k * 0.8), (cx - k * 0.2, cy + k * 0.4),
               (cx - k * 0.2, cy), (cx - k * 0.45, cy), (cx - k * 0.45, cy - k * 0.75)]
        draw.line(pts, fill=rgba, width=w, joint="round")
    elif kind in ("chart_bars", "chart"):
        bw = k * 0.36
        draw.rounded_rectangle([cx - k * 0.85, cy + k * 0.1, cx - k * 0.85 + bw, cy + k * 0.85], radius=int(bw * 0.4), fill=rgba)
        draw.rounded_rectangle([cx - k * 0.85 + bw + k * 0.2, cy - k * 0.35, cx - k * 0.85 + bw * 2 + k * 0.2, cy + k * 0.85], radius=int(bw * 0.4), fill=rgba)
        draw.rounded_rectangle([cx - k * 0.85 + bw * 2 + k * 0.4, cy - k * 0.85, cx - k * 0.85 + bw * 3 + k * 0.4, cy + k * 0.85], radius=int(bw * 0.4), fill=rgba)
    elif kind in ("target", "gptzero"):
        draw.ellipse([cx - k * 0.9, cy - k * 0.9, cx + k * 0.9, cy + k * 0.9], outline=rgba, width=w)
        draw.ellipse([cx - k * 0.52, cy - k * 0.52, cx + k * 0.52, cy + k * 0.52], outline=rgba, width=w)
        draw.ellipse([cx - k * 0.18, cy - k * 0.18, cx + k * 0.18, cy + k * 0.18], fill=rgba)
    elif kind == "turnitin":
        draw.line([(cx - k * 0.55, cy - k * 0.8), (cx + k * 0.15, cy - k * 0.8),
                   (cx + k * 0.55, cy - k * 0.4), (cx + k * 0.55, cy + k * 0.8),
                   (cx - k * 0.55, cy + k * 0.8), (cx - k * 0.55, cy - k * 0.8)],
                  fill=rgba, width=w, joint="round")
        draw.line([(cx + k * 0.15, cy - k * 0.8), (cx + k * 0.15, cy - k * 0.4),
                   (cx + k * 0.55, cy - k * 0.4)], fill=rgba, width=w, joint="round")
        draw.ellipse([cx - k * 0.22, cy - k * 0.08, cx + k * 0.16, cy + k * 0.3], outline=rgba, width=w)
        line([(cx + k * 0.16, cy + k * 0.1), (cx + k * 0.16, cy + k * 0.5)])
    elif kind == "originality":
        draw.arc([cx - k * 0.85, cy - k * 0.7, cx + k * 0.7, cy + k * 0.7], start=120, end=340, fill=rgba, width=w)
        draw.arc([cx - k * 0.85, cy - k * 0.2, cx + k * 0.85, cy + k * 0.85], start=30, end=180, fill=rgba, width=w)
        draw.arc([cx - k * 0.55, cy - k * 0.5, cx + k * 0.35, cy + k * 0.5], start=90, end=270, fill=rgba, width=w)
        draw.line([(cx - k * 0.05, cy - k * 0.6), (cx - k * 0.05, cy + k * 0.7)], fill=rgba, width=w)
        draw.arc([cx - k * 0.05, cy - k * 0.5, cx + k * 0.65, cy + k * 0.4], start=270, end=90, fill=rgba, width=w)
    elif kind in ("copyright", "copyleaks"):
        draw.ellipse([cx - k * 0.9, cy - k * 0.9, cx + k * 0.9, cy + k * 0.9], outline=rgba, width=w)
        draw.arc([cx - k * 0.5, cy - k * 0.5, cx + k * 0.5, cy + k * 0.5], start=45, end=315, fill=rgba, width=w)
    elif kind == "sun":
        draw.ellipse([cx - k * 0.45, cy - k * 0.45, cx + k * 0.45, cy + k * 0.45], outline=rgba, width=w)
        for i in range(8):
            a = math.pi * i / 4.0
            line([(cx + math.cos(a) * k * 0.65, cy + math.sin(a) * k * 0.65), (cx + math.cos(a) * k * 0.95, cy + math.sin(a) * k * 0.95)])
    elif kind == "chevron_down":
        line([(cx - k * 0.6, cy - k * 0.25), (cx, cy + k * 0.35), (cx + k * 0.6, cy - k * 0.25)])
    elif kind == "win_min":
        line([(cx - k * 0.7, cy), (cx + k * 0.7, cy)])
    elif kind == "win_max":
        draw.rectangle([cx - k * 0.65, cy - k * 0.65, cx + k * 0.65, cy + k * 0.65], outline=rgba, width=w)
    elif kind == "win_close":
        line([(cx - k * 0.65, cy - k * 0.65), (cx + k * 0.65, cy + k * 0.65)])
        line([(cx + k * 0.65, cy - k * 0.65), (cx - k * 0.65, cy + k * 0.65)])

    final_img = img.resize((max(1, int(s)), max(1, int(s))), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(final_img, master=master)
    _IMAGE_CACHE[key] = (photo, final_img)
    return photo


def draw_icon(c, kind, cx, cy, s, color, width=1.4):
    """Draw vector icon using anti-aliased PIL engine onto canvas."""
    photo = get_icon_photo(kind, s, color, master=c, width=width)
    c.create_image(cx, cy, image=photo, anchor="center")
    photos = getattr(c, "_icon_photos", None)
    if photos is None:
        photos = []
        c._icon_photos = photos
    photos.append(photo)


# ------------------------------------------------------- page background
class GlowBackground(tk.Canvas):
    """Uniform soft page canvas."""

    def __init__(self, master, **kw):
        super().__init__(master, highlightthickness=0, bd=0, bg=INK, **kw)
        self.bind("<Configure>", lambda e: self._paint())

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        self.delete("all")
        self.create_rectangle(0, 0, w, h, fill=INK, outline="")


# --------------------------------------------------------- rounded card
class RoundedCard(tk.Frame):
    """Clean white card with smooth corners and soft diffused shadow."""

    def __init__(self, master, radius=18, inset=0, shadow=8, fill=TRAY,
                 core=CORE, border=HAIRLINE, core_border=None, sunken=False,
                 parent_bg=None, **kw):
        super().__init__(master, bg=parent_bg or INK, **kw)
        self._r = radius
        self._shadow = shadow
        self._fill = fill
        self._border = border
        self._sunken = sunken
        self._pbg = parent_bg or INK
        self.canvas = tk.Canvas(self, width=1, height=1, highlightthickness=0,
                                bd=0, bg=self._pbg)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        pad = shadow + inset
        self.inner = tk.Frame(self, bg=core)
        self.inner.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.bind("<Configure>", lambda e: self.after_idle(self._paint))
        self.after_idle(self._paint)

    def _paint(self):
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        c.delete("all")
        photo = get_rounded_rect_img(
            w, h, self._r, fill=self._fill, outline=self._border,
            outline_width=1, parent_bg=self._pbg, shadow=self._shadow,
            shadow_dy=2, shadow_color=(100, 116, 139, 30), master=c
        )
        c.create_image(0, 0, image=photo, anchor="nw")
        self._card_photo = photo


# ------------------------------------------------------------- icon label
class IconLabel(tk.Frame):
    """Section header: vector icon + tracked uppercase title."""

    def __init__(self, master, icon, text, size=10, color=TEXT, bg=None,
                 icon_color=None, gap=10, **kw):
        bg = bg or master.cget("bg")
        super().__init__(master, bg=bg, **kw)
        self.canvas = tk.Canvas(self, width=22, height=22, bg=bg,
                                highlightthickness=0, bd=0)
        self.canvas.pack(side="left", padx=(0, gap))
        self.canvas.after_idle(
            lambda: draw_icon(self.canvas, icon, 11, 11, 16, icon_color or color)
        )
        label(self, tracked(text.upper()), size=size, color=color,
              weight="bold", bg=bg).pack(side="left")


# ------------------------------------------------------------ glyph button
class GlyphButton(tk.Canvas):
    """Small square icon button with hover feedback."""

    def __init__(self, master, icon, command=None, size=26, color="#64748b",
                 hover=ACCENT, bg=None, plate=False, **kw):
        self._bg = bg or INK
        super().__init__(master, width=size, height=size, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self._icon = icon
        self._size = size
        self._color = color
        self._hover_c = hover
        self._plate = plate
        self.command = command
        self._hover = False
        self.configure(cursor="hand2")
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", lambda e: self._set(True))
        self.bind("<Leave>", lambda e: self._set(False))
        self.bind("<Button-1>", self._click)
        self.after_idle(self._paint)

    def _set(self, v):
        if v != self._hover:
            self._hover = v
            self._paint()

    def _click(self, e):
        if self.command:
            self.command()

    def set_icon(self, icon):
        self._icon = icon
        self._paint()

    def _paint(self):
        s = self._size
        self.delete("all")
        if self._hover and self._plate:
            photo = get_rounded_rect_img(s, s, 6, fill=mix(self._bg, ACCENT, 0.08),
                                         parent_bg=self._bg, master=self)
            self.create_image(0, 0, image=photo, anchor="nw")
            self._ref = photo
        draw_icon(self, self._icon, s / 2.0, s / 2.0, 15,
                  self._hover_c if self._hover else self._color)


# --------------------------------------------------------- slim scrollbar
class SlimScrollbar(tk.Canvas):
    """Minimal rounded scrollbar for the text editors."""

    WIDTH = 8

    def __init__(self, master, target, bg="#ffffff", **kw):
        super().__init__(master, width=self.WIDTH, bg=bg, highlightthickness=0,
                         bd=0, **kw)
        self._bg = bg
        self.target = target
        self._first, self._last = 0.0, 1.0
        self._grab = None
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: setattr(self, "_grab", None))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        try:
            target.configure(yscrollcommand=self._sync)
        except Exception:
            pass

    def _sync(self, first, last):
        self._first, self._last = float(first), float(last)
        self._paint()

    def _geom(self, w, h):
        f, l = self._first, self._last
        if l - f >= 0.9995:
            return None
        top = 2 + f * (h - 4)
        bot = 2 + l * (h - 4)
        if bot - top < 24:
            mid = (top + bot) / 2
            top, bot = mid - 12, mid + 12
        return top, bot

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 10:
            return
        self.delete("all")
        g = self._geom(w, h)
        if not g:
            return
        top, bot = g
        hot = self._grab is not None
        c_fill = mix(self._bg, "#8fa2bd", 0.6 if not hot else 0.45)
        sw = w - 4
        sh = bot - top
        photo = get_rounded_rect_img(sw, sh, sw // 2, fill=c_fill, parent_bg=self._bg, master=self)
        self.create_image(2, top, image=photo, anchor="nw")
        self._ref = photo

    def _motion(self, e):
        g = self._geom(self.winfo_width(), self.winfo_height())
        if g and g[0] <= e.y <= g[1] and self._grab is None:
            self.configure(cursor="hand2")
        else:
            self.configure(cursor="")

    def _press(self, e):
        h = self.winfo_height()
        g = self._geom(self.winfo_width(), h)
        if not g:
            return
        top, bot = g
        if top <= e.y <= bot:
            self._grab = e.y - top
        else:
            self._grab = (bot - top) / 2
            self._move(e.y)
        self._paint()

    def _move(self, y):
        h = max(1, self.winfo_height())
        g = self._geom(self.winfo_width(), h)
        if not g:
            return
        top, bot = g
        frac = (y - self._grab - top) / max(1.0, (h - 4) - (bot - top))
        try:
            self.target.yview_moveto(max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def _drag(self, e):
        if self._grab is not None:
            self._move(e.y)


# ------------------------------------------------------------------- chip
class Chip(tk.Canvas):
    """Status pill with colored dot, label, and trailing chevron."""

    def __init__(self, master, text="Likely human", color=GOOD, height=30, bg=None,
                 size=9, trailing="chevron_down", **kw):
        self._bg = bg or "#ffffff"
        super().__init__(master, height=height, width=1, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self._text = text
        self._color = color
        self._size = size
        self._trailing = trailing
        self.bind("<Configure>", lambda e: self._paint())
        self.after_idle(self._autosize)
        self.after_idle(self._paint)

    def _autosize(self):
        tw = Fonts.measure(self._size, "bold")(self._text)
        w = 14 + 8 + 8 + tw + (20 if self._trailing else 12)
        self.configure(width=max(w, 115))

    def set(self, text, color=GOOD, pulse=False, trailing="chevron_down"):
        self._text = text
        self._color = color
        self._trailing = trailing
        self._autosize()
        self._paint()

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 10:
            return
        self.delete("all")
        r = int(h / 2)

        if self._color == GOOD:
            fill_c = "#ffffff"
            border_c = HAIRLINE
            fg_c = "#059669"
            dot_c = GOOD
        elif self._color == WARN:
            fill_c = "#fffbeb"
            border_c = "#fde68a"
            fg_c = "#d97706"
            dot_c = WARN
        elif self._color == BAD:
            fill_c = "#fef2f2"
            border_c = "#fecaca"
            fg_c = "#dc2626"
            dot_c = BAD
        else:
            fill_c = "#f8fafc"
            border_c = HAIRLINE
            fg_c = MUTED
            dot_c = FAINT

        photo = get_rounded_rect_img(w, h, r, fill=fill_c, outline=border_c,
                                     outline_width=1, parent_bg=self._bg, master=self)
        self.create_image(0, 0, image=photo, anchor="nw")
        self._chip_photo = photo

        cy = h / 2.0
        # Dot
        self.create_oval(14 - 3.5, cy - 3.5, 14 + 3.5, cy + 3.5,
                         fill=dot_c, outline="")
        # Text
        self.create_text(26, cy, text=self._text, anchor="w", fill=fg_c,
                         font=(Fonts.get("text"), self._size, "bold"))
        # Trailing chevron
        if self._trailing:
            draw_icon(self, self._trailing, w - 14, cy, 11, fg_c, width=1.5)


# ------------------------------------------------------ segmented control
class SegmentedControl(tk.Canvas):
    """Pill segmented control: recessed capsule track with royal blue active pill."""

    def __init__(self, master, options, value=None, on_change=None, height=44,
                 bg=None, radius=22, **kw):
        self._bg = bg or "#ffffff"
        super().__init__(master, height=height, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self.options = options            # [(key, label), ...]
        self.value = value or options[0][0]
        self.on_change = on_change
        self._r = radius
        self._hover = None
        self._anim = None
        self._ind = None
        self._pressed = None
        self._fs = 10
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.after_idle(self._paint)

    def _slot(self, w, h, i):
        pad = 4
        seg = (w - pad * 2) / len(self.options)
        return pad + i * seg, seg, pad + 3, h - pad * 2 - 6

    def _target(self, w, h):
        i = [k for k, _ in self.options].index(self.value)
        return self._slot(w, h, i)

    def _index_at(self, x):
        w = self.winfo_width()
        pad = 4
        seg = (w - pad * 2) / len(self.options)
        return max(0, min(len(self.options) - 1, int((x - pad) / seg)))

    def _set_hover(self, i):
        if i != self._hover:
            self._hover = i
            self._paint()

    def _on_motion(self, e):
        self._set_hover(self._index_at(e.x))

    def _on_press(self, e):
        self._pressed = self._index_at(e.x)
        self._paint()

    def _on_release(self, e):
        i = self._index_at(e.x)
        self._pressed = None
        key = self.options[i][0]
        if key != self.value:
            self.set_value(key, notify=True)
        else:
            self._paint()

    def set_value(self, key, notify=False):
        if key == self.value:
            return
        self.value = key
        self._spring_to_target()
        if notify and self.on_change:
            self.on_change(key)

    def _spring_to_target(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or self._ind is None:
            self._paint()
            return
        tx, tw, ty, th = self._target(w, h)
        self._anim = {"t": 0.0, "sx": self._ind[0], "sw": self._ind[1],
                      "tx": tx, "tw": tw}
        self._tick()

    def _tick(self):
        a = self._anim
        if not a:
            return
        a["t"] += 0.08
        t = min(1.0, a["t"])
        e = max(0.0, ease_out_back(t) if t < 1 else 1.0)
        self._ind = [a["sx"] + (a["tx"] - a["sx"]) * e,
                     a["sw"] + (a["tw"] - a["sw"]) * e]
        self._paint()
        if t < 1.0:
            self.after(16, self._tick)
        else:
            self._ind = [a["tx"], a["tw"]]
            self._anim = None
            self._paint()

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 10:
            return
        self.delete("all")
        # Recessed capsule track
        photo_track = get_rounded_rect_img(
            w, h, self._r, fill="#f1f5f9", outline="#e2e8f0",
            outline_width=1, parent_bg=self._bg, master=self
        )
        self.create_image(0, 0, image=photo_track, anchor="nw")
        self._track_photo = photo_track

        tx, tw, ty, th = self._target(w, h)
        if self._ind is None:
            self._ind = [tx, tw]
        ix, iw = self._ind
        rr = int(th / 2)

        # Hover on unselected
        for i, (key, text) in enumerate(self.options):
            if key != self.value and self._hover == i:
                sx, sw, sy, sh = self._slot(w, h, i)
                p_hov = get_rounded_rect_img(int(sw), int(sh), int(sh / 2),
                                             fill="#ffffff", outline="#e2e8f0",
                                             outline_width=1, parent_bg="#f1f5f9", master=self)
                self.create_image(int(sx), int(sy), image=p_hov, anchor="nw")

        # Active blue pill indicator
        hovered_sel = (self._hover is not None and
                       self.options[self._index_at(ix + iw / 2)][0] == self.value)
        top_c = mix(ACCENT_TOP, "#ffffff", 0.1) if hovered_sel else ACCENT_TOP
        photo_ind = get_rounded_rect_img(
            int(iw), int(th), rr, fill=(top_c, ACCENT_BOT),
            parent_bg="#f1f5f9", shadow=2, shadow_dy=1.5,
            shadow_color=(37, 99, 235, 75), master=self
        )
        self.create_image(int(ix), int(ty), image=photo_ind, anchor="nw")
        self._ind_photo = photo_ind

        # Option text + dots
        fnt_on = (Fonts.get("text"), self._fs, "bold")
        fnt_off = (Fonts.get("text"), self._fs)
        measure = Fonts.measure(self._fs)
        dot_r, dot_gap = 3.5, 8
        for i, (key, text) in enumerate(self.options):
            sx, sw, sy, sh = self._slot(w, h, i)
            on = key == self.value
            tw = measure(text)
            group = dot_r * 2 + dot_gap + tw
            gx = sx + (sw - group) / 2.0
            dot_c = "#ffffff" if on else ("#94a3b8" if self._hover == i else "#cbd5e1")
            cy = h / 2.0
            self.create_oval(gx, cy - dot_r, gx + dot_r * 2, cy + dot_r,
                             fill=dot_c, outline="")
            self.create_text(gx + dot_r * 2 + dot_gap + tw / 2.0, cy,
                             text=text, anchor="center",
                             fill="#ffffff" if on else (TEXT if self._hover == i else MUTED),
                             font=fnt_on if on else fnt_off)


# ----------------------------------------------------------------- switch
class Switch(tk.Canvas):
    """iOS-style smooth animated toggle switch."""

    W, H = 52, 30

    def __init__(self, master, value=False, on_change=None, bg=None, **kw):
        self._bg = bg or "#ffffff"
        super().__init__(master, width=self.W, height=self.H, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self.value = bool(value)
        self.on_change = on_change
        self._hover = False
        self._anim = None
        self._k = 1.0 if value else 0.0
        self.configure(cursor="hand2")
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", lambda e: self._h(True))
        self.bind("<Leave>", lambda e: self._h(False))
        self.bind("<Button-1>", lambda e: self.toggle())
        self.after_idle(self._paint)

    def _h(self, v):
        if v != self._hover:
            self._hover = v
            self._paint()

    def toggle(self):
        self.set(not self.value, notify=True)

    def set(self, value, notify=False):
        value = bool(value)
        if value == self.value and self._anim is None:
            return
        self.value = value
        self._anim = {"t": 0.0, "from": self._k, "to": 1.0 if value else 0.0}
        self._tick()
        if notify and self.on_change:
            self.on_change(value)

    def _tick(self):
        a = self._anim
        if not a:
            return
        a["t"] += 0.10
        t = min(1.0, a["t"])
        self._k = a["from"] + (a["to"] - a["from"]) * ease_out_cubic(t)
        self._paint()
        if t < 1.0:
            self.after(16, self._tick)
        else:
            self._k = a["to"]
            self._anim = None
            self._paint()

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        self.delete("all")
        r = int(h / 2)
        if self.value:
            fill_val = (ACCENT_TOP, ACCENT_BOT) if self._hover else (ACCENT, ACCENT_BOT)
            outline_val = None
        else:
            fill_val = "#cbd5e1" if self._hover else "#e2e8f0"
            outline_val = "#94a3b8" if self._hover else "#cbd5e1"

        photo_track = get_rounded_rect_img(
            w, h, r, fill=fill_val, outline=outline_val, outline_width=1,
            parent_bg=self._bg, shadow=1 if self.value else 0,
            shadow_dy=1, shadow_color=(37, 99, 235, 60), master=self
        )
        self.create_image(0, 0, image=photo_track, anchor="nw")
        self._track_photo = photo_track

        # Knob
        kr = int(r - 3)
        kx = 3 + kr + (w - 6 - 2 * kr) * self._k
        cy = h / 2.0
        photo_knob = get_rounded_rect_img(
            kr * 2, kr * 2, kr, fill="#ffffff", outline="#e2e8f0",
            outline_width=1, parent_bg=fill_val if isinstance(fill_val, str) else ACCENT,
            shadow=2, shadow_dy=1, shadow_color=(0, 0, 0, 45), master=self
        )
        self.create_image(int(kx - kr), int(cy - kr), image=photo_knob, anchor="nw")
        self._knob_photo = photo_knob


# ------------------------------------------------------------- pill button
class PillButton(tk.Canvas):
    """Button widget: primary CTA with gradient & arrow, or clean white secondary button."""

    def __init__(self, master, text, command=None, kind="soft", icon=None,
                 icon_side="left", trailing=None, height=42, radius=12,
                 parent_bg=None, size=10, **kw):
        self._pbg = parent_bg or INK
        super().__init__(master, width=1, height=height, bg=self._pbg,
                         highlightthickness=0, bd=0, **kw)
        self._text = text
        self.command = command
        self.kind = kind
        self.icon = icon
        self.icon_side = icon_side
        self.trailing = trailing
        self._r = radius
        self._hover = False
        self._press = False
        self._disabled = False
        self._size = size
        self.configure(cursor="hand2")
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", lambda e: self._h(True))
        self.bind("<Leave>", lambda e: (self._h(False), self._press_set(False)))
        self.bind("<ButtonPress-1>", lambda e: self._press_set(True))
        self.bind("<ButtonRelease-1>", self._release)
        self.after_idle(self._paint)

    def _h(self, v):
        if v != self._hover and not self._disabled:
            self._hover = v
            self.configure(cursor="hand2" if v else "")
            self._paint()

    def _press_set(self, v):
        self._press = v
        self._paint()

    def _release(self, e):
        self._press_set(False)
        if self._disabled:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if 0 <= e.x <= w and 0 <= e.y <= h and self.command:
            self.command()

    def set_disabled(self, disabled=True):
        self._disabled = disabled
        self.configure(cursor="" if disabled else "hand2")
        self._paint()

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 10:
            return
        self.delete("all")
        lift = 1 if self._press else 0

        if self.kind == "primary":
            if self._disabled:
                fill = ("#8fb0dd", "#6f93c9")
                sh = 0
            elif self._press:
                fill = ("#1d4ed8", "#1e40af")
                sh = 1
            elif self._hover:
                fill = ("#3b82f6", "#2563eb")
                sh = 4
            else:
                fill = (ACCENT_TOP, ACCENT_BOT)
                sh = 3
            photo = get_rounded_rect_img(
                w, h, self._r, fill=fill, parent_bg=self._pbg,
                shadow=sh, shadow_dy=2, shadow_color=(37, 99, 235, 75), master=self
            )
            self.create_image(0, lift, image=photo, anchor="nw")
            self._btn_photo = photo
            fg = "#ffffff"
        else:
            if self._disabled:
                fill = "#f8fafc"
                outline = HAIRLINE
                fg = "#b9c3d2"
                sh = 0
            elif self._press:
                fill = "#f1f5f9"
                outline = HAIRLINE_2
                fg = TEXT
                sh = 0
            elif self._hover:
                fill = "#f8fafc"
                outline = "#cbd5e1"
                fg = TEXT
                sh = 2
            else:
                fill = "#ffffff"
                outline = "#e2e8f0"
                fg = "#334155"
                sh = 1
            photo = get_rounded_rect_img(
                w, h, self._r, fill=fill, outline=outline, outline_width=1,
                parent_bg=self._pbg, shadow=sh, shadow_dy=1,
                shadow_color=(100, 116, 139, 25), master=self
            )
            self.create_image(0, lift, image=photo, anchor="nw")
            self._btn_photo = photo

        # Text and icons
        tw = Fonts.measure(self._size, "bold")(self._text)
        icon_w = 24 if self.icon else 0
        trail_w = 32 if self.trailing else 0
        cy = h / 2.0 + lift

        if self.trailing:
            # Primary button with centered content and right arrow disc
            tx = (w - trail_w) / 2.0 + 8
            ix = tx - tw / 2.0 - 12
            draw_icon(self, self.icon, ix, cy, 15, fg, width=1.4)
            self.create_text(tx, cy, text=self._text, fill=fg,
                             font=(Fonts.get("text"), self._size, "bold"))
            # Disc on right with dark blue arrow
            dx = w - 24
            dr = 12
            disc_photo = get_rounded_rect_img(
                dr * 2, dr * 2, dr, fill="#60a5fa",
                parent_bg=fill[1] if isinstance(fill, (list, tuple)) else fill,
                master=self
            )
            self.create_image(dx - dr, cy - dr, image=disc_photo, anchor="nw")
            self._disc_photo = disc_photo
            draw_icon(self, self.trailing, dx, cy, 12, "#1d4ed8", width=1.6)
        elif self.icon:
            # Secondary 2x2 buttons
            if self.icon_side == "left":
                content_w = icon_w + tw
                start = (w - content_w) / 2.0
                ix = start + 8
                tx = start + icon_w + tw / 2.0
            else:
                content_w = tw + icon_w
                start = (w - content_w) / 2.0
                tx = start + tw / 2.0
                ix = start + tw + 14
            draw_icon(self, self.icon, ix, cy, 14, fg, width=1.3)
            self.create_text(tx, cy, text=self._text, fill=fg,
                             font=(Fonts.get("text"), self._size, "bold"))
        else:
            self.create_text(w / 2.0, cy, text=self._text, fill=fg,
                             font=(Fonts.get("text"), self._size, "bold"))


# ------------------------------------------------------------ status toast
class StatusToast(tk.Canvas):
    """Auto-hiding pill for action feedback."""

    def __init__(self, master, text="", parent_bg=None, height=36, **kw):
        self._pbg = parent_bg or INK
        super().__init__(master, width=1, height=height, bg=self._pbg,
                         highlightthickness=0, bd=0, **kw)
        self._text = text
        self._hide_job = None
        self.bind("<Configure>", lambda e: self._paint())
        self.after_idle(self._paint)

    def flash(self, text, ms=2400):
        self._text = text
        self._autosize()
        self._paint()
        if self._hide_job:
            try:
                self.after_cancel(self._hide_job)
            except Exception:
                pass
        self._hide_job = self.after(ms, self.hide)

    def start_busy(self, text):
        self._text = text
        self._autosize()
        self._paint()

    def stop_busy(self, text):
        self.flash(text, ms=3000)

    def hide(self):
        self._text = ""
        self.configure(width=1)
        self.delete("all")

    def _autosize(self):
        if not self._text:
            self.configure(width=1)
            return
        tw = Fonts.measure(9, "bold")(self._text)
        self.configure(width=tw + 36)

    def _paint(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or not self._text:
            self.delete("all")
            return
        self.delete("all")
        r = int(h / 2)
        photo = get_rounded_rect_img(
            w, h, r, fill="#ffffff", outline="#e2e8f0", outline_width=1,
            parent_bg=self._pbg, shadow=3, shadow_dy=1.5,
            shadow_color=(100, 116, 139, 40), master=self
        )
        self.create_image(0, 0, image=photo, anchor="nw")
        self._toast_photo = photo
        self.create_text(w / 2.0, h / 2.0, text=self._text, anchor="center",
                         fill=TEXT, font=(Fonts.get("text"), 9, "bold"))


# ------------------------------------------------------------ metric tile
class MetricTile(tk.Frame):
    """Detector-style readout: icon badge + label row, value right, green progress bar."""

    HEIGHT = 68

    def __init__(self, master, title, icon="target", value="12%", bar=0.12,
                 bar_color=GOOD, bg=None, **kw):
        bg = bg or "#ffffff"
        super().__init__(master, bg=bg, **kw)
        self.canvas = tk.Canvas(self, width=1, height=self.HEIGHT, bg=bg,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._title = title
        self._icon = icon
        self._value = value
        self._bar = float(bar)
        self._bar_color = bar_color
        self._hover = False
        self.canvas.bind("<Configure>", lambda e: self._paint())
        self.canvas.bind("<Motion>", lambda e: self._h(True))
        self.canvas.bind("<Leave>", lambda e: self._h(False))
        self.after_idle(self._paint)

    def _h(self, v):
        if v != self._hover:
            self._hover = v
            self._paint()

    def set(self, value=None, bar=None, color=None):
        if value is not None:
            self._value = str(value)
        if bar is not None:
            self._bar = float(bar)
        if color is not None:
            self._bar_color = color
        self._paint()

    def _paint(self):
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 40 or h < 24:
            return
        c.delete("all")
        bg_card = "#fbfdff" if self._hover else "#f8fafc"
        outline = "#cbd5e1" if self._hover else "#e2e8f0"
        photo = get_rounded_rect_img(
            w, h, 12, fill=bg_card, outline=outline, outline_width=1,
            parent_bg="#ffffff", shadow=1, shadow_dy=1,
            shadow_color=(100, 116, 139, 20), master=c
        )
        c.create_image(0, 0, image=photo, anchor="nw")
        self._tile_photo = photo

        mid = 24
        # Draw dark slate icon badge matching design reference
        draw_icon(c, self._icon, 24, mid, 18, "#334155", width=1.5)
        # Title
        c.create_text(42, mid, text=self._title, anchor="w", fill=TEXT,
                      font=(Fonts.get("text"), 10, "bold"))
        # Value (e.g. 12%)
        c.create_text(w - 18, mid, text=self._value, anchor="e", fill=TEXT,
                      font=(Fonts.get("text"), 10, "bold"))

        # Progress bar
        by = h - 20
        bw = w - 36
        bar_h = 7
        photo_track = get_rounded_rect_img(
            bw, bar_h, int(bar_h / 2), fill="#e2e8f0", parent_bg=bg_card, master=c
        )
        c.create_image(18, by, image=photo_track, anchor="nw")
        self._track_photo = photo_track

        if self._bar > 0:
            fw = max(bar_h, int(bw * max(0.0, min(1.0, self._bar))))
            photo_fill = get_rounded_rect_img(
                fw, bar_h, int(bar_h / 2), fill=self._bar_color,
                parent_bg=bg_card, master=c
            )
            c.create_image(18, by, image=photo_fill, anchor="nw")
            self._fill_photo = photo_fill


# ------------------------------------------------------- intro reveal
class IntroReveal:
    """Mask reveal transition on window open."""

    def __init__(self, master):
        pass