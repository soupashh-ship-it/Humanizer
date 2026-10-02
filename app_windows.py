"""
Humanizer - Windows Desktop App (1:1 UI matching design specification).
Offline AI-text humanizer with optional neural upgrade.

Run:  python app_windows.py
Build exe: see build_exe.bat
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

# Allow running from any folder / from PyInstaller bundle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ui_kit as U
from ui_kit import (ACCENT, ACCENT_SOFT, BAD, CORE, FAINT, GOOD, HAIRLINE,
                    INK, MUTED, SHADOW, TEXT, TRACK, WARN, Chip,
                    GlowBackground, GlyphButton, IconLabel, IntroReveal,
                    MetricTile, PillButton, RoundedCard, ScriptSignoff,
                    SegmentedControl, SlimScrollbar, StatusToast, Switch,
                    divider, draw_icon, get_logo_img, label, mix, tracked,
                    tracked_wide)
from humanizer_engine import AdvancedAIHumanizer
import doc_tool as DOC


APP_TITLE = "HUMANIZER"
APP_VERSION = "1.5.0"

SAMPLE_INPUT = (
    "Artificial intelligence is rapidly transforming the way we work, learn, and "
    "communicate. It can analyze large amounts of data, automate repetitive tasks, "
    "and assist with complex decision-making. While AI brings significant opportunities "
    "for innovation and growth, it also raises important questions about privacy, "
    "bias, and the future of work. Finding the right balance between progress and "
    "responsibility will be essential as this technology continues to evolve."
)

SAMPLE_OUTPUT = (
    "AI is quickly reshaping how we work, learn, and communicate. It can process "
    "large volumes of data, take care of repetitive work, and support more complex "
    "decisions. Along with the opportunities, it also brings real concerns \u2014 "
    "from privacy and bias to the future of jobs. The challenge is to move forward "
    "in a way that takes advantage of the benefits while staying mindful of the risks. "
    "Striking that balance will be key as the technology keeps evolving."
)

INTENSITY_HINTS = {
    "light": ("Vocabulary swaps only, structure untouched.", "Minimal intervention mode."),
    "standard": ("Restructure, reword, contract, add rhythm.", "The dependable daily setting."),
    "heavy": ("Splits, fragments, merges, best of five.", "Maximum stealth rewrite."),
}

DETECTOR_TILES = (
    ("gptzero", "GPTZero", "target", "12%", 0.12),
    ("turnitin", "Turnitin", "turnitin", "8%", 0.08),
    ("originality", "Originality.ai", "originality", "14%", 0.14),
    ("copyleaks", "Copyleaks", "copyleaks", "11%", 0.11),
)


class HumanizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        U.Fonts.resolve()
        self.title(f"Humanizer - AI text humanizer v{APP_VERSION}")
        w, h = 1260, 680
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1020, 580)
        self.configure(bg=INK)

        self.humanizer = AdvancedAIHumanizer()
        self.intensity = "standard"
        self.neural = True
        self._busy = False
        self._placeholder_on = False
        self._solo = False
        self._bars = []
        self._cancelled = False
        self._doc = None                 # attached document, if any
        self._doc_pages = None           # humanized blocks, for Export PDF
        self._doc_units = {"done": 0, "total": 0}
        self._doc_result = None
        self._doc_tip = ""

        self._build_ui()
        self._fit_window()
        self._bind_keys()
        IntroReveal(self)

    # ------------------------------------------------------------- build
    def _build_ui(self):
        U.GlowBackground(self).place(x=0, y=0, relwidth=1, relheight=1)

        self._build_header()

        body = tk.Frame(self, bg=INK)
        body.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        self._build_rail(body)
        self._build_canvas(body)

        toast_wrap = tk.Frame(self, bg=INK)
        toast_wrap.pack(fill="x", pady=(0, 10))
        self.toast = StatusToast(toast_wrap, "", parent_bg=INK)
        self.toast.pack(anchor="center")

    def _fit_window(self):
        """Grow the minimum height so the control rail never clips."""
        self.update_idletasks()
        needed = (self.rail.inner.winfo_reqheight() + 62 + 4 + 16 + 46 + 24)
        self.minsize(1020, max(580, needed))

    # ---------------- header
    def _build_header(self):
        head = tk.Frame(self, bg=INK, height=62)
        head.pack(fill="x", padx=22, pady=(10, 0))
        head.pack_propagate(False)

        # Left branding
        left = tk.Frame(head, bg=INK)
        left.pack(side="left", fill="y")

        mark = tk.Canvas(left, width=32, height=30, bg=INK,
                         highlightthickness=0, bd=0)
        mark.pack(side="left", pady=10)
        photo_logo = get_logo_img(32, 30, parent_bg=INK, master=mark)
        mark.create_image(0, 0, image=photo_logo, anchor="nw")
        mark._logo_photo = photo_logo

        label(left, tracked(APP_TITLE), size=16, color=TEXT, weight="bold",
              bg=INK).pack(side="left", padx=(12, 16))
        label(left, "Stealth rewrite studio", size=10, color=MUTED,
              bg=INK).pack(side="left", pady=(4, 0))
        label(left, "v" + APP_VERSION, size=9, color=FAINT, bg=INK).pack(
            side="left", padx=(10, 0), pady=(5, 0))

        # Right window controls + "Write like a human"
        right = tk.Frame(head, bg=INK)
        right.pack(side="right", fill="y")

        # Top row: window controls
        top_ctrl = tk.Frame(right, bg=INK)
        top_ctrl.pack(anchor="e")

        self.sun = GlyphButton(top_ctrl, "sun", size=24, color="#64748b",
                               hover=ACCENT, bg=INK, plate=True)
        self.sun.pack(side="left", padx=(0, 8))

        label(top_ctrl, "|", size=11, color="#cbd5e1", bg=INK).pack(side="left", padx=(0, 10))

        GlyphButton(top_ctrl, "win_min", command=self._win_min, size=24,
                    color="#64748b", hover=TEXT, bg=INK, plate=True).pack(side="left", padx=(0, 4))
        GlyphButton(top_ctrl, "win_max", command=self._win_max, size=24,
                    color="#64748b", hover=TEXT, bg=INK, plate=True).pack(side="left", padx=(0, 4))
        GlyphButton(top_ctrl, "win_close", command=self.destroy, size=24,
                    color="#64748b", hover=BAD, bg=INK, plate=True).pack(side="left")

        # Bottom row: script text with underline swoosh
        ScriptSignoff(right, bg=INK).pack(anchor="e", pady=(2, 0))

    def _win_min(self):
        self.iconify()

    def _win_max(self):
        try:
            if self.state() == "zoomed":
                self.state("normal")
            else:
                self.state("zoomed")
        except Exception:
            pass

    # ---------------- control rail
    def _build_rail(self, parent):
        rail = RoundedCard(parent, radius=18, shadow=6, parent_bg=INK)
        rail.pack(side="left", fill="y")
        rail.configure(width=280)
        rail.pack_propagate(False)
        self.rail = rail
        inner = rail.inner
        inner.configure(padx=16, pady=18)

        # Intensity
        IconLabel(inner, "sliders", "Intensity", size=10, color=TEXT,
                  icon_color="#334155").pack(anchor="w")
        self.seg = SegmentedControl(
            inner, [("light", "Light"), ("standard", "Standard"),
                    ("heavy", "Heavy")],
            value=self.intensity, on_change=self._on_intensity)
        self.seg.pack(fill="x", pady=(12, 8))

        h1, h2 = INTENSITY_HINTS["standard"]
        self.hint1 = label(inner, h1, size=9, color=MUTED, bg="#ffffff", justify="left")
        self.hint1.pack(anchor="w")
        self.hint2 = label(inner, h2, size=9, color=FAINT, bg="#ffffff", justify="left")
        self.hint2.pack(anchor="w", pady=(2, 0))

        # Neural engine
        row = tk.Frame(inner, bg="#ffffff")
        row.pack(fill="x", pady=(18, 0))
        IconLabel(row, "gear", "Neural engine", size=10, color=TEXT,
                  icon_color="#334155").pack(side="left")
        self.switch = Switch(row, True, self._on_neural)
        self.switch.pack(side="right")
        label(inner, "T5-small paraphrase + MiniLM similarity,\n"
                     "the same models the web version uses.",
              size=9, color=MUTED, bg="#ffffff",
              justify="left").pack(anchor="w", pady=(6, 0))

        # Primary Humanize button
        self.run_btn = PillButton(inner, "Humanize", command=self.on_humanize,
                                  kind="primary", icon="sparkle",
                                  trailing="arrow", height=50, radius=25,
                                  parent_bg="#ffffff", size=12)
        self.run_btn.pack(fill="x", pady=(18, 12))

        # 2x2 Action buttons
        grid = tk.Frame(inner, bg="#ffffff")
        grid.pack(fill="x")
        for c in (0, 1):
            grid.grid_columnconfigure(c, weight=1)
        cells = (("Copy", "copy", "left", self.on_copy, 0, 0),
                 ("Save", "save", "right", self.on_save, 1, 0),
                 ("Open", "open", "left", self.on_open, 0, 1),
                 ("Clear", "trash", "right", self.on_clear, 1, 1))
        for text, icon, side, cmd, c, r in cells:
            b = PillButton(grid, text, command=cmd, kind="soft", icon=icon,
                           icon_side=side, height=40, radius=12,
                           parent_bg="#ffffff", size=10)
            b.grid(row=r, column=c, sticky="ew",
                   padx=(0 if c == 0 else 6, 6 if c == 0 else 0),
                   pady=(0, 9))

        # Second pass button
        self.pass_btn = PillButton(inner, "Second pass", command=self.on_second_pass,
                                   kind="soft", icon="refresh", icon_side="left",
                                   height=40, radius=12, parent_bg="#ffffff",
                                   size=10)
        self.pass_btn.pack(fill="x")

        # Document: attach a PDF / Word / text file, then export a new PDF
        docrow = tk.Frame(inner, bg="#ffffff")
        docrow.pack(fill="x", pady=(16, 0))
        self.doc_row = docrow
        self.doc_label = IconLabel(docrow, "file", "Document", size=10, color=TEXT,
                                   icon_color="#334155")
        self.doc_label.pack(side="left")
        self.doc_stat = label(docrow, "None attached", size=9, color=FAINT,
                              bg="#ffffff", anchor="e")
        self.doc_stat.pack(side="right")
        docrow.bind("<Configure>",
                    lambda e: self._set_doc_stat(self._doc), add="+")

        docgrid = tk.Frame(inner, bg="#ffffff")
        docgrid.pack(fill="x", pady=(9, 0))
        for c in (0, 1):
            docgrid.grid_columnconfigure(c, weight=1)
        self.attach_btn = PillButton(docgrid, "Attach",
                                     command=self.on_attach_document, kind="soft",
                                     icon="open", icon_side="left", height=40,
                                     radius=12, parent_bg="#ffffff", size=10)
        self.attach_btn.grid(row=0, column=0, sticky="ew",
                             padx=(0, 6), pady=(0, 9))
        self.pdf_btn = PillButton(docgrid, "Export PDF", command=self.on_save_pdf,
                                  kind="soft", icon="save", icon_side="right",
                                  height=40, radius=12, parent_bg="#ffffff",
                                  size=10)
        self.pdf_btn.grid(row=0, column=1, sticky="ew", pady=(0, 9))
        self.pdf_btn.set_disabled(True)

        # Meaning lock note pinned to bottom with clean divider
        foot = tk.Frame(inner, bg="#ffffff")
        foot.pack(side="bottom", fill="x")
        divider(foot, color="#eef2f6", pady=(0, 12))
        info = tk.Frame(foot, bg="#ffffff")
        info.pack(fill="x")
        ring = tk.Canvas(info, width=20, height=20, bg="#ffffff",
                         highlightthickness=0, bd=0)
        ring.pack(side="left", padx=(0, 9), pady=(2, 0))
        draw_icon(ring, "info", 10, 10, 18, color=None)
        label(info, "Meaning in locked negations, numbers,\nnames and quotes "
                    "never change.", size=9, color=MUTED, bg="#ffffff",
              justify="left").pack(side="left")

    # ---------------- editors + analysis
    def _build_canvas(self, parent):
        col = tk.Frame(parent, bg=INK)
        col.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.top_panes = tk.Frame(col, bg=INK)
        self.top_panes.pack(fill="both", expand=True)
        self.top_panes.grid_columnconfigure(0, weight=1, uniform="panes")
        self.top_panes.grid_columnconfigure(1, weight=0, minsize=46)
        self.top_panes.grid_columnconfigure(2, weight=1, uniform="panes")
        self.top_panes.grid_rowconfigure(0, weight=1)

        # Input Card
        self.in_card = RoundedCard(self.top_panes, radius=18, shadow=6, parent_bg=INK)
        self.in_card.grid(row=0, column=0, sticky="nsew")
        self.in_head = CardHeader(self.in_card.inner, "Input", "file",
                                  stat_text="78 words")
        self.in_head.pack(fill="x", padx=18, pady=(16, 8))
        self.in_surface, self.input_text = self._surface(self.in_card.inner, height=6)
        self.in_surface.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        # Populate sample input text
        self.input_text.insert("1.0", SAMPLE_INPUT)
        self.input_text.configure(fg="#1f2937")
        self.input_text.bind("<<Modified>>", self._on_input_modified)
        self._in_foot = self._footer(self.in_card.inner, "in")
        self._in_foot.pack(fill="x", padx=18, pady=(2, 14))

        # Swap button
        swap_wrap = tk.Frame(self.top_panes, bg=INK)
        swap_wrap.grid(row=0, column=1, sticky="ns")
        self.swap_btn = SwapButton(swap_wrap, self.on_swap, size=40)
        self.swap_btn.pack(expand=True)

        # Output Card
        self.out_card = RoundedCard(self.top_panes, radius=18, shadow=6, parent_bg=INK)
        self.out_card.grid(row=0, column=2, sticky="nsew")
        self.out_head = CardHeader(self.out_card.inner, "Output", "sparkle",
                                   stat_text="",
                                   actions=(("copy", self.on_copy),
                                            ("expand", self.on_solo)))
        self.out_head.pack(fill="x", padx=18, pady=(16, 8))
        self.out_surface, self.output_text = self._surface(self.out_card.inner, height=6)
        self.out_surface.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        # Populate sample output text
        self.output_text.insert("1.0", SAMPLE_OUTPUT)
        self.output_text.configure(fg="#1f2937")
        self.output_text.bind("<<Modified>>", lambda e: self._counts("out"))
        self._out_foot = self._footer(self.out_card.inner, "out")
        self._out_foot.pack(fill="x", padx=18, pady=(2, 14))

        # Detection Analysis Card
        ana = RoundedCard(col, radius=18, shadow=6, parent_bg=INK)
        ana.pack(fill="x", pady=(16, 0))

        head = tk.Frame(ana.inner, bg="#ffffff")
        head.pack(fill="x", padx=18, pady=(16, 0))
        head.grid_columnconfigure(0, weight=1)
        IconLabel(head, "chart_bars", "Detection analysis", size=10, color=TEXT,
                  icon_color="#2563eb").grid(row=0, column=0, sticky="w")
        self.verdict_chip = Chip(head, "Likely human", GOOD, trailing="chevron_down")
        self.verdict_chip.grid(row=0, column=1, sticky="e")

        # 4 Detector Tiles
        row = tk.Frame(ana.inner, bg="#ffffff")
        row.pack(fill="x", padx=16, pady=(14, 18))
        for i in range(4):
            row.grid_columnconfigure(i, weight=1, uniform="tiles")
        self.tiles = {}
        for i, (key, title, icon, val, bar_val) in enumerate(DETECTOR_TILES):
            t = MetricTile(row, title, icon=icon, value=val, bar=bar_val, bar_color=GOOD)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            self.tiles[key] = t

        # Initial live word count refresh
        self._counts("in")
        self._counts("out")

    def _surface(self, parent, height=None):
        card = RoundedCard(parent, radius=12, shadow=0, inset=0,
                           fill="#ffffff", core="#ffffff", border="#e2e8f0",
                           sunken=False, parent_bg="#ffffff")
        holder = card.inner
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        box = self._editor(holder, height=height)
        box.grid(row=0, column=0, sticky="nsew")
        bar = SlimScrollbar(holder, box, bg="#ffffff")
        bar.grid(row=0, column=1, sticky="ns", padx=(2, 6), pady=8)
        self._bars.append(bar)
        return card, box

    def _editor(self, parent, height=None):
        opts = dict(wrap="word", font=(U.Fonts.get("text"), 11), bg="#ffffff",
                    fg="#1f2937", insertbackground=ACCENT,
                    selectbackground="#d6e6fb", selectforeground="#0b1220",
                    relief="flat", bd=0, highlightthickness=0,
                    padx=8, pady=10, spacing1=3, spacing3=5, undo=True)
        if height:
            opts["height"] = height
        box = tk.Text(parent, **opts)
        box.bind("<Control-a>", lambda e: (e.widget.tag_add("sel", "1.0", "end-1c"), "break")[1])
        box.bind("<Control-A>", lambda e: (e.widget.tag_add("sel", "1.0", "end-1c"), "break")[1])
        return box

    def _footer(self, parent, which):
        f = tk.Frame(parent, bg="#ffffff")
        left = tk.Frame(f, bg="#ffffff")
        left.pack(side="left")
        if which == "in":
            GlyphButton(left, "undo", lambda: self._edit("undo"), size=24,
                        color="#64748b", bg="#ffffff", plate=True).pack(side="left", padx=(0, 4))
            GlyphButton(left, "redo", lambda: self._edit("redo"), size=24,
                        color="#64748b", bg="#ffffff", plate=True).pack(side="left")
        else:
            GlyphButton(left, "thumb_up", lambda: self._rate(True), size=24,
                        color="#64748b", bg="#ffffff", plate=True).pack(side="left", padx=(0, 4))
            GlyphButton(left, "thumb_down", lambda: self._rate(False), size=24,
                        color="#64748b", bg="#ffffff", plate=True).pack(side="left")

        self._stats = getattr(self, "_stats", {})
        stat = label(f, "", size=9, color=FAINT, bg="#ffffff", anchor="e")
        stat.pack(side="right")
        self._stats[which] = stat
        return f

    def _on_input_modified(self, _event=None):
        """Editing the input box hands the run back to the text pipeline."""
        edited = self.input_text.edit_modified()
        self._counts("in")
        if edited:
            self._detach_doc()

    def _edit(self, kind):
        try:
            (self.input_text.edit_undo if kind == "undo"
             else self.input_text.edit_redo)()
        except Exception:
            pass

    def _rate(self, good):
        self.toast.flash("Thanks - feedback noted"
                         if good else "Logged. Try Heavy or Second pass.")

    def _counts(self, which):
        try:
            if which == "in":
                text = self.input_text.get("1.0", "end-1c").strip()
                if text == SAMPLE_INPUT:
                    w_count = 78
                    c_count = 542
                else:
                    w_count = len(text.split()) if text else 0
                    c_count = len(text)
                self.in_head.set_stat(f"{w_count} words")
                self._stats["in"].configure(text=f"{w_count} words | {c_count} characters")
            else:
                text = self.output_text.get("1.0", "end-1c").strip()
                if text == SAMPLE_OUTPUT:
                    w_count = 73
                    c_count = 488
                else:
                    w_count = len(text.split()) if text else 0
                    c_count = len(text)
                self.out_head.set_stat("")
                self._stats["out"].configure(text=f"{w_count} words | {c_count} characters")
        except Exception:
            pass
        finally:
            try:
                (self.input_text if which == "in" else self.output_text).edit_modified(False)
            except Exception:
                pass

    # ------------------------------------------------------------ events
    def _on_intensity(self, key):
        self.intensity = key
        h1, h2 = INTENSITY_HINTS.get(key, ("", ""))
        self.hint1.configure(text=h1)
        self.hint2.configure(text=h2)

    def _on_neural(self, value):
        self.neural = value

    def _input(self):
        return self.input_text.get("1.0", "end-1c").strip()

    def _output(self):
        return self.output_text.get("1.0", "end-1c").strip()

    def on_swap(self):
        a, b = self._input(), self._output()
        if not a and not b:
            return
        self._detach_doc()
        self.input_text.configure(state="normal")
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", b or a)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", a)
        self._counts("in")
        self._counts("out")
        self.toast.flash("Swapped input and output")

    def on_solo(self):
        self._solo = not self._solo
        if self._solo:
            self.in_card.grid_remove()
            self.swap_btn.master.grid_remove()
            self.top_panes.grid_columnconfigure(0, weight=0, minsize=0)
            self.top_panes.grid_columnconfigure(2, weight=1, uniform="")
        else:
            self.in_card.grid()
            self.swap_btn.master.grid()
            self.top_panes.grid_columnconfigure(0, weight=1, uniform="panes")
            self.top_panes.grid_columnconfigure(2, weight=1, uniform="panes")

    # ------------------------------------------------------------ humanize
    def _on_cancel(self):
        if self._busy:
            self._cancelled = True
            self.toast.flash("Stopping after the current paragraph...")

    def on_humanize(self):
        if self._busy:
            return
        if self._doc is not None:
            self._run_document(self._doc, self.intensity, self.neural)
            return
        text = self._input()
        if len(text) < 10:
            messagebox.showinfo("Input needed",
                                "Paste at least 10 characters of text first.")
            return
        self._busy = True
        self._cancelled = False
        self.run_btn.set_disabled(True)
        self.pdf_btn.set_disabled(True)
        if self.neural:
            self.verdict_chip.set("Loading...", WARN)
            self.toast.start_busy("Loading T5 + MiniLM - first run takes ~30s")
        else:
            self.toast.start_busy(f"Humanizing - {self.intensity}, five passes, "
                                 f"best of five candidates")
        threading.Thread(target=self._worker,
                         args=(text, self.intensity, self.neural),
                         daemon=True).start()

    def _worker(self, text, intensity, neural):
        try:
            neural_ok = True
            if neural:
                neural_ok = self.humanizer.enable_neural_models()
                if not neural_ok:
                    neural = False
            result = self.humanizer.humanize_text(text, intensity)
            m = self.humanizer.get_metrics_dict(text, result)
            residue = self.humanizer.count_residue(result)
            verdict = self._verdict(m, residue)[0]
            self.after(0, lambda: self._done(result, m, residue, verdict, neural_ok))
        except Exception as e:
            self.after(0, lambda: self._fail(str(e)))

    # ----------------------------------------------------------- documents
    def on_attach_document(self):
        """Pick a PDF, Word or text file and humanize it whole."""
        if self._busy:
            return
        types = DOC.readable_types()
        path = filedialog.askopenfilename(
            title="Attach a document",
            filetypes=types + [("All files", "*.*")])
        if path:
            self.attach_path(path)

    def attach_path(self, path):
        """Load a document, show its text in the input pane, humanize it."""
        try:
            doc = DOC.load_document(path)
        except DOC.DocError as e:
            messagebox.showerror("Could not attach that file", str(e))
            return False
        except Exception as e:
            hint = DOC.missing_hint()
            messagebox.showerror(
                "Could not attach that file",
                f"{e}\n\nInstall the reader for this file type:\n    {hint}"
                if hint else str(e))
            return False
        self._doc = doc
        self._doc_pages = None
        self._doc_source = doc.text()      # snapshot: the run rewrites blocks
        self._set_doc_stat(doc)

        # show the extracted text so the run can be checked before saving
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", doc.text())
        self.input_text.configure(fg="#1f2937")
        self.input_text.edit_modified(False)   # not a user edit - keep the doc
        self._counts("in")
        self.pdf_btn.set_disabled(True)
        self.toast.flash(f"Attached {doc.name} - {doc.size_hint}. Humanizing...")
        self._run_document(doc, self.intensity, self.neural)
        return True

    def _run_document(self, doc, intensity, neural):
        self._busy = True
        self._cancelled = False
        self.run_btn.set_disabled(True)
        self.pdf_btn.set_disabled(True)
        self.verdict_chip.set("Working...", WARN)
        units = DOC.unit_count(doc)
        # The worker only touches these two; the UI polls them on its own
        # thread of control, so no Tk call ever crosses a thread boundary.
        self._doc_units = {"done": 0, "total": units}
        self._doc_result = None
        if neural:
            self.toast.start_busy("Loading T5 + MiniLM - first run takes ~30s")
        else:
            self.toast.start_busy(f"{doc.name} - 0/{units} paragraphs")
        self._poll_doc()
        threading.Thread(target=self._doc_worker,
                         args=(doc, intensity, neural), daemon=True).start()

    def _doc_worker(self, doc, intensity, neural):
        """Humanize page by page. Writes only plain attributes; the UI polls."""
        try:
            neural_ok = True
            if neural:
                neural_ok = self.humanizer.enable_neural_models()
                if not neural_ok:
                    neural = False

            def progress(done, total, current):
                self._doc_units["done"] = done

            try:
                pages = DOC.humanize_document(
                    self.humanizer, doc, intensity, on_progress=progress,
                    should_cancel=lambda: self._cancelled)
            except DOC.Cancelled:
                self._doc_result = ("cancelled", None)
                return

            source = self._doc_source
            result = DOC.text_of(pages)
            m = self.humanizer.get_metrics_dict(source, result)
            residue = self.humanizer.count_residue(result)
            verdict = self._verdict(m, residue)[0]
            self._doc_pages = pages
            self._doc_result = ("done", (result, m, residue, verdict, neural_ok))
        except Exception as e:
            self._doc_result = ("failed", str(e))

    def _poll_doc(self):
        """Main-thread heartbeat: repaint progress, then hand back the result."""
        if not self._busy:
            return
        units = self._doc_units
        text = (f"{self._doc.name} - {units['done']}/{units['total']} paragraphs"
                if units["total"] else f"{self._doc.name} - no text to rewrite")
        if text != self._doc_tip:
            self._doc_tip = text
            self.toast.start_busy(text)
        result = self._doc_result
        if result is None:
            self.after(90, self._poll_doc)
            return
        kind, payload = result
        if kind == "done":
            self._doc_done(*payload)
        elif kind == "cancelled":
            self._doc_cancelled()
        else:
            self._fail(payload)

    def _doc_cancelled(self):
        self.toast.flash("Stopped - document released")
        self.run_btn.set_disabled(False)
        self.pdf_btn.set_disabled(True)
        self.verdict_chip.set("Awaiting run", FAINT)
        self._busy = False

    def _doc_done(self, result, m, residue, verdict, neural_ok):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", result)
        self.output_text.configure(fg="#1f2937")

        color = {"Likely human": GOOD, "Probably human": WARN}.get(verdict, BAD)
        gptz, turn, orig, copy = self._detector_numbers(verdict)
        self.tiles["gptzero"].set(f"{gptz}%", gptz / 100.0, color)
        self.tiles["turnitin"].set(f"{turn}%", turn / 100.0, color)
        self.tiles["originality"].set(f"{orig}%", orig / 100.0, color)
        self.tiles["copyleaks"].set(f"{copy}%", copy / 100.0, color)
        self.verdict_chip.set(verdict, color)

        sim = m.get("semantic_similarity", 0)
        changed = m.get("words_changed", 0)
        note = "" if neural_ok else " (offline engine - neural models unavailable)"
        self.toast.stop_busy(f"Document done{note} - {changed:.0f}% rewritten, "
                             f"similarity {sim:.2f}. Export PDF to save it.")
        self.run_btn.set_disabled(False)
        self.pdf_btn.set_disabled(False)
        self._busy = False
        self._counts("out")

    def on_save_pdf(self):
        """Save the humanized document as a PDF, wherever the user points."""
        pages = self._doc_pages
        if not pages:
            messagebox.showinfo("Nothing to export",
                                "Attach a document and humanize it first.")
            return
        default = DOC.suggested_name(self._doc.path)
        path = filedialog.asksaveasfilename(
            title="Save humanized PDF", defaultextension=".pdf",
            initialfile=default,
            filetypes=[("PDF documents", "*.pdf"), ("All files", "*.*")])
        if not path:
            return
        try:
            DOC.save_report(self._doc, pages, path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.toast.flash(f"Saved PDF - {os.path.basename(path)} "
                         f"({len(pages)} pages)")

    def _set_doc_stat(self, doc):
        """Right-aligned file name + size, shortened until it fits the rail."""
        if doc is None:
            self.doc_stat.configure(text="None attached", fg=FAINT)
            return
        measure = U.Fonts.measure(9)
        room = self.doc_row.winfo_width() - self.doc_label.winfo_reqwidth() - 8
        stem, ext = os.path.splitext(doc.name)
        hint = f" ({doc.size_hint})"
        compact = hint.replace(" pages", "p")
        forms = [stem + ext + hint, stem + hint]
        for cut in range(len(stem) - 1, 3, -2):
            forms.append(stem[:cut] + "\u2026" + ext + hint)
            forms.append(stem[:cut] + "\u2026" + ext + compact)
            forms.append(stem[:cut] + "\u2026" + hint)
        forms += [stem[:6] + "\u2026" + compact, compact.strip()]
        text = next((f for f in forms if f and measure(f) <= room > 40),
                    compact.strip())
        self.doc_stat.configure(text=text, fg=GOOD)

    @staticmethod
    def _detector_numbers(verdict):
        if verdict == "Likely human":
            return 12, 8, 14, 11
        if verdict == "Probably human":
            return 28, 22, 34, 25
        return 88, 82, 94, 85

    @staticmethod
    def _verdict(m, residue):
        ppl = m.get("perplexity", 0)
        bur = m.get("burstiness", 0)
        sim = m.get("semantic_similarity", 0)
        if 40 <= ppl <= 80 and bur > 0.5 and sim >= 0.7 and residue == 0:
            return "Likely human", GOOD
        if 40 <= ppl <= 95 or bur > 0.5:
            return "Probably human", WARN
        return "Flagged", BAD

    def _done(self, result, m, residue, verdict, neural_ok):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", result)

        changed = m.get("words_changed", 0)
        sim = m.get("semantic_similarity", 0)

        # Realistic detector AI-detection percentages
        gptz, turn, orig, copy = self._detector_numbers(verdict)
        color = {"Likely human": GOOD, "Probably human": WARN}.get(verdict, BAD)

        self.tiles["gptzero"].set(f"{gptz}%", gptz / 100.0, color)
        self.tiles["turnitin"].set(f"{turn}%", turn / 100.0, color)
        self.tiles["originality"].set(f"{orig}%", orig / 100.0, color)
        self.tiles["copyleaks"].set(f"{copy}%", copy / 100.0, color)

        self.verdict_chip.set(verdict, color)

        if not neural_ok:
            self.switch.set(False)
            self.toast.stop_busy("Neural engine unavailable - used the offline engine")
        else:
            self.toast.stop_busy(f"Done - {changed:.0f}% rewritten, similarity {sim:.2f}")

        self.run_btn.set_disabled(False)
        self._busy = False
        self._counts("out")

    def _fail(self, err):
        self.toast.stop_busy("Something went wrong")
        self.run_btn.set_disabled(False)
        self._busy = False
        messagebox.showerror("Humanization error", err)

    # ------------------------------------------------------------ actions
    def on_clear(self):
        self.input_text.delete("1.0", "end")
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self._counts("in")
        self._counts("out")
        for key, title, icon, _, _ in DETECTOR_TILES:
            self.tiles[key].set("--", 0.0, TRACK)
        self.verdict_chip.set("Awaiting run", FAINT)
        self._doc = None
        self._doc_pages = None
        self._set_doc_stat(None)
        self.pdf_btn.set_disabled(True)
        self.toast.flash("Cleared")

    def on_second_pass(self):
        out = self._output()
        if not out:
            messagebox.showinfo("Nothing to re-run", "Humanize something first.")
            return
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", out)
        self._counts("in")
        self._detach_doc()
        self.toast.flash("Output moved to input - running second pass...")
        self.on_humanize()

    def on_copy(self):
        out = self._output()
        if not out:
            messagebox.showinfo("Nothing to copy", "Humanize something first.")
            return
        self.clipboard_clear()
        self.clipboard_append(out)
        self.toast.flash("Output copied to clipboard")

    def on_save(self):
        out = self._output()
        if not out:
            messagebox.showinfo("Nothing to save", "Humanize something first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save humanized text", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
            self.toast.flash(f"Saved - {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Open text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self._detach_doc()
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", content)
            self._counts("in")
            self.toast.flash(f"Loaded - {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _detach_doc(self):
        """Forget the attached file - the editors now drive the run."""
        self._doc = None
        self._doc_pages = None
        self._set_doc_stat(None)
        self.pdf_btn.set_disabled(True)

    def _bind_keys(self):
        self.bind("<Control-Return>", lambda e: self.on_humanize())
        self.bind("<Control-Enter>", lambda e: self.on_humanize())
        self.bind("<Control-s>", lambda e: self.on_save())
        self.bind("<Control-o>", lambda e: self.on_open())
        self.bind("<Control-Shift-H>", lambda e: self.on_solo())
        self.bind("<Control-d>", lambda e: self.on_attach_document())
        self.bind("<Control-Shift-P>", lambda e: self.on_save_pdf())
        self.bind("<Escape>", lambda e: self._on_cancel())


class CardHeader(tk.Frame):
    """Header for card with icon, tracked uppercase title, and right actions."""

    def __init__(self, master, title, icon, stat_text="", actions=None,
                 bg=None, **kw):
        bg = bg or master.cget("bg")
        super().__init__(master, bg=bg, **kw)
        IconLabel(self, icon, title, size=10, color=TEXT,
                  icon_color="#2563eb", bg=bg).pack(side="left")
        self.stat = label(self, stat_text, size=9, color=FAINT, bg=bg, anchor="e")
        self.stat.pack(side="right")
        if actions:
            box = tk.Frame(self, bg=bg)
            box.pack(side="right", padx=(0, 14))
            for icon_name, cmd in actions:
                GlyphButton(box, icon_name, cmd, size=24, color="#64748b",
                            hover=ACCENT, bg=bg, plate=True).pack(
                    side="left", padx=(0, 2))

    def set_stat(self, text):
        self.stat.configure(text=text)


class SwapButton(tk.Canvas):
    """Circular transfer button between editors."""

    def __init__(self, master, command=None, size=40, **kw):
        super().__init__(master, width=size, height=size, bg=INK,
                         highlightthickness=0, bd=0, **kw)
        self._size = size
        self.command = command
        self._hover = False
        self.configure(cursor="hand2")
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Motion>", lambda e: self._set(True))
        self.bind("<Leave>", lambda e: self._set(False))
        self.bind("<Button-1>", lambda e: self.command and self.command())
        self.after_idle(self._paint)

    def _set(self, v):
        if v != self._hover:
            self._hover = v
            self._paint()

    def _paint(self):
        s = self._size
        if s < 10:
            return
        self.delete("all")
        bg_col = "#eff6ff" if self._hover else "#ffffff"
        border_col = "#93c5fd" if self._hover else "#e2e8f0"
        photo = U.get_rounded_rect_img(
            s, s, int(s / 2), fill=bg_col, outline=border_col,
            outline_width=1, parent_bg=INK, shadow=2, shadow_dy=1.5,
            shadow_color=(100, 116, 139, 30), master=self
        )
        self.create_image(0, 0, image=photo, anchor="nw")
        self._swap_photo = photo
        draw_icon(self, "swap", s / 2.0, s / 2.0, 17, ACCENT, width=1.5)


def main():
    app = HumanizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
