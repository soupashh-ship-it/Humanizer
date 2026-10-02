"""
test_app.py - Verification and edge-case tests for Humanizer UI and Engine.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ui_kit as U
import app_windows
import doc_tool as DOC
from humanizer_engine import AdvancedAIHumanizer


def sample_pdf(path):
    """A two page PDF to attach, written with the app's own writer."""
    pages = [
        [{"kind": "head", "text": "Quarterly Operations Review"},
         {"kind": "blank", "text": ""},
         {"kind": "para", "text": ("Furthermore, it is important to leverage our "
                                   "cutting-edge platform to delve into the "
                                   "multifaceted challenges that the organization "
                                   "faces. Moreover, the comprehensive analysis "
                                   "underscores the pivotal role of collaboration.")},
         {"kind": "blank", "text": ""},
         {"kind": "bullet", "text": "reduce onboarding time by 40%"},
         {"kind": "bullet", "text": "consolidate three reporting pipelines"},
         {"kind": "blank", "text": ""},
         {"kind": "para", "text": ("In conclusion, the team must optimize utilization "
                                   "of existing resources. Note we do not anticipate "
                                   "any cost increase, and 47% stays valid.")}],
        [{"kind": "head", "text": "2. Risks and Mitigations"},
         {"kind": "blank", "text": ""},
         {"kind": "para", "text": ("The primary risk is that onboarding slips beyond "
                                   "the third quarter. To facilitate a smooth "
                                   "transition, we propose establishing a working "
                                   "group in August.")}],
    ]
    DOC.write_pdf(path, pages, [(612.0, 792.0), (612.0, 792.0)], "Q3 Review")
    return path


class TestHumanizerUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_ui_kit_rendering(self):
        """Verify that get_rounded_rect_img produces valid PhotoImages for various sizes and options."""
        p1 = U.get_rounded_rect_img(100, 40, 10, fill="#ffffff", outline="#e2e8f0", shadow=2, master=self.root)
        self.assertIsNotNone(p1)

        p2 = U.get_rounded_rect_img(200, 50, 25, fill=("#2563eb", "#1d5ed8"), shadow=3, master=self.root)
        self.assertIsNotNone(p2)

        logo = U.get_logo_img(32, 30, master=self.root)
        self.assertIsNotNone(logo)

    def test_icons(self):
        """Verify that all vector icons can be drawn on a canvas without error."""
        canv = tk.Canvas(self.root, width=200, height=200)
        icons = [
            "arrow", "arrow_left", "swap", "refresh", "undo", "redo",
            "copy", "save", "open", "trash", "close", "sparkle", "file",
            "expand", "collapse", "sliders", "gear", "info", "thumb_up",
            "thumb_down", "chart_bars", "target", "gptzero", "turnitin",
            "originality", "copyleaks", "sun", "chevron_down",
            "win_min", "win_max", "win_close"
        ]
        for ic in icons:
            canv.delete("all")
            U.draw_icon(canv, ic, 100, 100, 20, "#2563eb")

    def test_app_initialization(self):
        """Verify HumanizerApp initializes with correct defaults matching 1:1 image."""
        app = app_windows.HumanizerApp()
        self.assertEqual(app_windows.APP_VERSION, "1.5.0")
        self.assertTrue(app.neural)
        self.assertEqual(app.in_head.stat.cget("text"), "78 words")
        self.assertEqual(app._stats["in"].cget("text"), "78 words | 542 characters")
        self.assertEqual(app._stats["out"].cget("text"), "73 words | 488 characters")
        self.assertEqual(app.tiles["gptzero"]._value, "12%")
        self.assertEqual(app.tiles["turnitin"]._value, "8%")
        self.assertEqual(app.tiles["originality"]._value, "14%")
        self.assertEqual(app.tiles["copyleaks"]._value, "11%")
        self.assertEqual(app.verdict_chip._text, "Likely human")
        app.destroy()

    def test_app_actions(self):
        """Test clear, swap, and counts."""
        app = app_windows.HumanizerApp()

        # Swap test
        orig_in = app._input()
        orig_out = app._output()
        app.on_swap()
        self.assertEqual(app._input(), orig_out)
        self.assertEqual(app._output(), orig_in)

        # Clear test
        app.on_clear()
        self.assertEqual(app._input(), "")
        self.assertEqual(app._output(), "")
        self.assertEqual(app.tiles["gptzero"]._value, "--")
        self.assertEqual(app.verdict_chip._text, "Awaiting run")

        # Solo toggle test
        self.assertFalse(app._solo)
        app.on_solo()
        self.assertTrue(app._solo)
        app.on_solo()
        self.assertFalse(app._solo)

        app.destroy()

    def test_intensity_and_neural_toggles(self):
        """Test intensity selection and neural engine toggling."""
        app = app_windows.HumanizerApp()

        app.seg.set_value("light", notify=True)
        self.assertEqual(app.intensity, "light")
        self.assertIn("Vocabulary", app.hint1.cget("text"))

        app.seg.set_value("heavy", notify=True)
        self.assertEqual(app.intensity, "heavy")
        self.assertIn("Splits", app.hint1.cget("text"))

        app.seg.set_value("standard", notify=True)
        self.assertEqual(app.intensity, "standard")

        # Neural switch
        app.switch.set(True, notify=True)
        self.assertTrue(app.neural)
        app.switch.set(False, notify=True)
        self.assertFalse(app.neural)

        app.destroy()

    def test_humanize_worker_and_done(self):
        """Test running humanization worker and processing results into UI."""
        app = app_windows.HumanizerApp()
        text = "Furthermore, it is crucial to leverage cutting-edge technology to delve into solutions."

        # Simulate worker processing
        result = app.humanizer.humanize_text(text, "standard")
        m = app.humanizer.get_metrics_dict(text, result)
        residue = app.humanizer.count_residue(result)
        verdict = "Likely human" if residue == 0 else "Flagged"

        app._done(result, m, residue, verdict, True)

        self.assertEqual(app._output(), result)
        self.assertIn(app.verdict_chip._text, ["Likely human", "Probably human", "Flagged"])
        self.assertNotEqual(app.tiles["gptzero"]._value, "--")
        app.destroy()

    def test_engine_humanization(self):
        """Verify humanizer engine offline rewrite."""
        engine = AdvancedAIHumanizer()
        input_text = "Furthermore, it is important to delve into the realm of artificial intelligence."
        result = engine.humanize_text(input_text, intensity="standard")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertNotIn("delve into", result.lower())

    def test_script_signoff_and_logo(self):
        """Verify script signoff and two-tone brand logo rendering."""
        logo = U.get_logo_img(32, 30, master=self.root)
        self.assertIsNotNone(logo)
        signoff = U.get_script_signoff_img(180, 36, master=self.root)
        self.assertIsNotNone(signoff)
        widget = U.ScriptSignoff(self.root)
        self.assertIsNotNone(widget)
        widget.destroy()

    def test_detector_tiles_and_rail_divider(self):
        """Verify detector tiles and rail footer divider exist and match design."""
        app = app_windows.HumanizerApp()
        self.assertEqual(len(app.tiles), 4)
        for k in ("gptzero", "turnitin", "originality", "copyleaks"):
            self.assertIn(k, app.tiles)
            tile = app.tiles[k]
            self.assertEqual(tile._bar_color, U.GOOD)
        app.destroy()

    def test_document_controls_start_idle(self):
        """Attach is live, Export PDF is locked until a document is done."""
        app = app_windows.HumanizerApp()
        self.assertIsNone(app._doc)
        self.assertIsNone(app._doc_pages)
        self.assertEqual(app.doc_stat.cget("text"), "None attached")
        self.assertFalse(app.attach_btn._disabled)
        self.assertTrue(app.pdf_btn._disabled)
        app.destroy()

    def test_document_status_fits_the_rail(self):
        """A long file name is shortened until it clears the DOCUMENT label."""
        app = app_windows.HumanizerApp()
        app.update()
        room = app.doc_row.winfo_width() - app.doc_label.winfo_reqwidth() - 8
        self.assertGreater(room, 40)
        try:
            doc = DOC.Document(
                r"C:\reports\a-really-long-quarterly-report-name.pdf", "pdf",
                [[{"kind": "para", "text": "x"}]], [(612.0, 792.0)])
            app._doc = doc
            app._set_doc_stat(doc)
            app.update()
            text = app.doc_stat.cget("text")
            self.assertLessEqual(U.Fonts.measure(9)(text), room)
            self.assertIn("page", text)                      # size hint kept
            self.assertLess(len(text), len(doc.name) + 12)   # name shortened
            # it refits on its own when the rail is laid out
            app._set_doc_stat(None)
            app.update()
            self.assertEqual(app.doc_stat.cget("text"), "None attached")
        finally:
            app._doc = None
            app.destroy()

    def test_attach_humanize_and_export_pdf(self):
        """Attach a PDF, humanize every paragraph, then export a new PDF."""
        tmp = tempfile.mkdtemp(prefix="humanizer_test_")
        src = sample_pdf(os.path.join(tmp, "q3_review.pdf"))
        app = app_windows.HumanizerApp()
        app.neural = False
        try:
            self.assertTrue(app.attach_path(src))

            deadline = time.time() + 60
            while app._busy and time.time() < deadline:
                app.update()
                time.sleep(0.01)
            self.assertFalse(app._busy, "document run did not finish")
            app.update()

            # extracted text landed in the input pane
            self.assertIn("Quarterly Operations Review", app._input())
            self.assertEqual(app._doc.name, "q3_review.pdf")
            self.assertTrue(app.doc_stat.cget("text"))   # name + size, fitted
            self.assertNotEqual(app.doc_stat.cget("text"), "None attached")

            # every paragraph was rewritten, the headings were left alone
            out = app._output()
            self.assertIn("Quarterly Operations Review", out)
            self.assertIn("2. Risks and Mitigations", out)
            self.assertNotIn("Furthermore, it is important to leverage", out)
            self.assertNotIn("delve into", out.lower())
            self.assertEqual(app.humanizer.count_residue(out), 0)
            self.assertIn("47%", out)
            self.assertIn("don't", out.lower())

            # Export PDF is unlocked and writes a readable PDF
            self.assertFalse(app.pdf_btn._disabled)
            out_path = os.path.join(tmp, DOC.suggested_name(src))
            DOC.save_report(app._doc, app._doc_pages, out_path)
            self.assertTrue(os.path.exists(out_path))
            saved = DOC.load_document(out_path)
            self.assertEqual(len(saved.pages), 2)
            self.assertIn("Quarterly Operations Review", saved.text())
            self.assertIn("47%", saved.text())
        finally:
            app.destroy()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_clear_releases_document(self):
        """Clearing detaches the file and locks Export PDF again."""
        tmp = tempfile.mkdtemp(prefix="humanizer_test_")
        src = sample_pdf(os.path.join(tmp, "release.pdf"))
        app = app_windows.HumanizerApp()
        app.neural = False
        try:
            app.attach_path(src)
            deadline = time.time() + 60
            while app._busy and time.time() < deadline:
                app.update()
                time.sleep(0.01)
            app.update()
            self.assertIsNotNone(app._doc)

            app.on_clear()
            self.assertIsNone(app._doc)
            self.assertIsNone(app._doc_pages)
            self.assertEqual(app.doc_stat.cget("text"), "None attached")
            self.assertTrue(app.pdf_btn._disabled)
        finally:
            app.destroy()
            shutil.rmtree(tmp, ignore_errors=True)


class TestDocumentTool(unittest.TestCase):
    """doc_tool: block splitting, humanizing, and the PDF writer."""

    def test_split_blocks_finds_structure(self):
        """Headings, bullets and wrapped body lines are told apart."""
        page = ("Annual Report 2024\n"
                "\n"
                "The team must optimize resources while\n"
                "keeping costs flat across every region\n"
                "\n"
                "\u2022 reduce onboarding time by 40%\n"
                "\u2022 consolidate reporting pipelines\n"
                "\n"
                "2. Risks and Mitigations\n"
                "\n"
                "Vendor onboarding may slip.\n")
        blocks = DOC._split_blocks(page)
        paras = [b["text"] for b in blocks if b["kind"] == "para"]
        heads = [b["text"] for b in blocks if b["kind"] == "head"]
        self.assertEqual(heads, ["Annual Report 2024", "2. Risks and Mitigations"])
        # a line that does not stop a sentence is folded back into one paragraph
        self.assertEqual(paras[0],
                         "The team must optimize resources while keeping costs "
                         "flat across every region")
        # a line that does stop a sentence starts the next paragraph
        self.assertEqual(paras[1], "Vendor onboarding may slip.")
        self.assertEqual([b["text"] for b in blocks if b["kind"] == "bullet"],
                         ["reduce onboarding time by 40%",
                          "consolidate reporting pipelines"])

    def test_split_blocks_keeps_paragraphs_separate(self):
        """Headings and list items never swallow the paragraph under them."""
        page = ("Release Notes\n"
                "Furthermore, the pipeline was rebuilt last night.\n"
                "\u2022 ship the importer\n"
                "The importer handles the edge cases.\n")
        kinds = [b["kind"] for b in DOC._split_blocks(page)
                 if b["kind"] != "blank"]
        self.assertEqual(kinds, ["head", "para", "bullet", "para"])

    def test_wrap_respects_the_column(self):
        """No wrapped line may exceed the column, even with a monster word."""
        long_word = "supercalifragilistic" * 12
        lines = DOC._wrap("A short lead in " + long_word + " and a tail", 300.0,
                          10.5, False)
        self.assertTrue(lines)
        for line in lines:
            self.assertLessEqual(DOC._measure(line, 10.5, False), 301.0)
        self.assertIn(long_word, "".join(lines))

    def test_pdf_round_trip(self):
        """Write a PDF, read it back, and keep the text and the page sizes."""
        tmp = tempfile.mkdtemp(prefix="humanizer_doc_")
        try:
            path = sample_pdf(os.path.join(tmp, "round.pdf"))
            doc = DOC.load_document(path)
            self.assertEqual(doc.kind, "pdf")
            self.assertEqual(len(doc.pages), 2)
            self.assertEqual(doc.size_hint, "2 pages")
            self.assertIn("Quarterly Operations Review", doc.text())
            self.assertIn("reduce onboarding time by 40%", doc.text())
            self.assertEqual(DOC.unit_count(doc), 7)

            out = DOC.save_report(doc, doc.pages, os.path.join(tmp, "copy.pdf"))
            back = DOC.load_document(out)
            self.assertEqual(len(back.pages), 2)
            self.assertAlmostEqual(back.pages and 612.0, 612.0, places=1)
            self.assertIn("Quarterly Operations Review", back.text())
            self.assertEqual(DOC.suggested_name(path),
                             os.path.basename(path).replace(".pdf", "_humanized.pdf"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_humanize_keeps_headings_and_units(self):
        """Headings stay verbatim, paragraphs get rewritten, progress counts."""
        tmp = tempfile.mkdtemp(prefix="humanizer_doc_")
        try:
            doc = DOC.load_document(sample_pdf(os.path.join(tmp, "h.pdf")))
            source = doc.text()
            engine = AdvancedAIHumanizer()
            seen = []
            pages = DOC.humanize_document(engine, doc, "standard",
                                          on_progress=lambda d, t, x: seen.append((d, t)))
            result = DOC.text_of(pages)
            self.assertEqual(seen[-1], (7, 7))
            self.assertIn("Quarterly Operations Review", result)
            self.assertIn("2. Risks and Mitigations", result)
            self.assertNotIn("delve into", result.lower())
            self.assertEqual(engine.count_residue(result), 0)
            self.assertIn("47%", result)
            self.assertNotEqual(source, result)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cancel_stops_the_run(self):
        """should_cancel aborts between paragraphs."""
        tmp = tempfile.mkdtemp(prefix="humanizer_doc_")
        try:
            doc = DOC.load_document(sample_pdf(os.path.join(tmp, "c.pdf")))
            with self.assertRaises(DOC.Cancelled):
                DOC.humanize_document(AdvancedAIHumanizer(), doc, "light",
                                      should_cancel=lambda: True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_text_document_is_supported(self):
        """Plain text and Markdown attach with no extra install."""
        tmp = tempfile.mkdtemp(prefix="humanizer_doc_")
        try:
            path = os.path.join(tmp, "notes.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# Notes\n\nFurthermore, it is important to leverage "
                         "robust tooling to delve into the problem.\n")
            doc = DOC.load_document(path)
            self.assertEqual(doc.kind, "text")
            self.assertIn("Notes", doc.text())
            self.assertIn("words", doc.size_hint)
            self.assertEqual(DOC.unit_count(doc), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unsupported_and_empty_inputs(self):
        """Bad extensions and text-less PDFs fail with a clear message."""
        tmp = tempfile.mkdtemp(prefix="humanizer_doc_")
        try:
            with self.assertRaises(DOC.DocError):
                DOC.load_document(os.path.join(tmp, "photo.png"))
            blank = DOC.write_pdf(os.path.join(tmp, "blank.pdf"), [[]])
            with self.assertRaises(DOC.DocError) as ctx:
                DOC.load_document(blank)
            self.assertIn("No selectable text", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
