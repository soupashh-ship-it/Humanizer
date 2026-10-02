# Humanizer - Windows App

Offline AI-text humanizer for Windows, rebuilt from your original `app.py` (Gradio version)
so it runs as a real desktop app with **zero heavy dependencies**.

## The app (v1.5)

Light studio interface - paper-white cards on a soft blue-grey page:

- **Header** - slanted brand mark, letterspaced `H U M A N I Z E R`, tagline,
  version `v1.5.0`, theme glyph, window controls, and a hand-written "Write like a human" sign-off
- **Control rail** - intensity as a spring-animated segmented pill with status
  dots, `N E U R A L  E N G I N E` switch, royal blue CTA with sparkle icon and a
  nested arrow disc, Copy / Save / Open / Clear, Second pass, a `D O C U M E N T`
  block (Attach + Export PDF), and a pinned meaning-lock note
- **Editors** - 46 / 54 split, rounded writing surfaces, document and sparkle card
  headers, undo / redo and thumb actions, live `words | characters` counters,
  and a circular transfer button between the two panes
- **Detection analysis** - verdict chip (`Likely human`) plus four detector tiles
  with emerald green progress bars: GPTZero, Turnitin, Originality.ai, Copyleaks
- Soft diffused shadows, hairline rings, staggered mask-reveal entrance,
  1px vector icons throughout, and an auto-hiding status pill for feedback

Every animation touches canvas geometry and colour only, so it stays smooth on
any machine. All colour lives in one palette at the top of `ui_kit.py`.

> **Honest metrics:** the four tiles are measured locally by the engine.
> No detection service (GPTZero, Turnitin, Copyleaks, Originality.ai) is ever
> contacted, so the app does not invent their scores.

### Attach a document (v1.5)

The **Document** block in the control rail handles whole files, not just the
clipboard:

1. **Attach** opens a file picker. PDFs, `.docx`, `.txt` and `.md` are all read
   (PDF via `pypdf`, Word via `python-docx` - both pip-installable, text needs
   nothing). The file name and size appear next to `D O C U M E N T` (shortened
   to fit), and the extracted text lands in the Input pane so you can check it
   before saving.
2. **Humanize** then runs automatically, **paragraph by paragraph**, and the
   status pill counts up (`report.pdf - 148/325 paragraphs`). Headings and
   list items keep their wording; everything else goes through the same 5-pass
   pipeline as pasted text, so `47%`, `do not`, names and quotes all survive.
   `Esc` stops a long run.
3. **Export PDF** unlocks when the run finishes and asks where to save. The app
   writes the humanized document as a new PDF - its own stdlib PDF writer, no
   extra install needed - keeping the source page size, headings, paragraphs and
   bullets, with justified body text. One source page starts one output page, so
   a page that outgrows its original simply continues onto the next.

Editing the Input pane, swapping, opening a text file, or **Clear** releases the
attached document, and the Humanize button goes back to driving the text boxes.

> Layout is reconstructed from the extracted text, so fonts, images and tables
> become text: images are dropped, table rows come out as `a | b | c`, and
> characters outside the base-14 font set (CJK, for instance) are replaced
> with `?`. Scanned PDFs have no text layer - run OCR first.

### Keyboard

| Shortcut | Action |
|---|---|
| `Ctrl + Enter` | Humanize |
| `Ctrl + S` | Save output as text |
| `Ctrl + O` | Open a text file |
| `Ctrl + D` | Attach a document |
| `Ctrl + Shift + P` | Save the humanized document as PDF |
| `Ctrl + A` | Select all in a box |
| `Ctrl + Shift + H` | Solo / restore the output pane |
| `Esc` | Stop a running document |

## Why a rebuild instead of packaging app.py directly?

Your original `app.py` needs: `torch`, `transformers`, `sentence-transformers`, `spacy`,
`nltk` model downloads, `gradio` web server. That means:

- 2-4 GB download, slow startup, needs internet for first-run model downloads
- Fragile on Windows (torch / CUDA / spacy model issues)
- Runs as a browser page, not a desktop app

The Windows version keeps **100% of the original humanization logic**:

- All 50+ AI-flagged phrase replacements (`delve into`, `furthermore`, `leverage`, ...)
- All 30+ contractions, human starters, fillers, natural transitions
- Same 5-pass pipeline: pattern elimination > restructure > vocabulary > human touches > polish
- Same metrics: perplexity, burstiness, Flesch readability, semantic similarity
- Same intensities: light / standard / heavy

What changed: T5/BERT/spacy/nltk replaced with built-in rule-based equivalents,
so the app is **instant, offline, and ~11 MB as an .exe**.

> **v1.2 neural option:** the exact T5-small + MiniLM models from the
> HuggingFace Space can be switched on with the **Neural engine** toggle
> (or `humanize_cli.py --neural`). Same paraphrase recipe as the site
> (`temperature=0.7, top_p=0.9`), plus guards the site lacks: T5 stutter /
> language-flip rejection, best-of candidate selection, and a zero-AI-vocabulary
> guarantee. Install once (internet needed, ~600 MB total):
>
> ```bat
> python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
> python -m pip install -r requirements_optional.txt
> ```
>
> Models download automatically on the first neural run, then work offline.
> Without them the app behaves exactly as before and says so.

> **Meaning lock (v1.2):** the engine never changes what your text says,
> only how it sounds. No strengthened claims (`shows` never becomes
> `proves`), no invented logic (sentence merges use `and`/`;` only, never
> `but`/`so`), and neural mode hard-rejects any paraphrase that drops a
> negation, number, name, or quote - falling back to a safe rewrite
> instead. Verified: `do not`/`cannot`, `47%`/`2023`/`NASA`, and
> `"quoted terms"` all survive Heavy in both modes.
>
> **Detector tip (v1.2):** if a detector still flags the output, switch on
> **Neural engine** and run **Heavy**, or humanize twice
> (output -> input again). The engine guarantees zero known AI-vocabulary
> in every output and automatically keeps the stealthiest candidate.

## Files

| File | Purpose |
|---|---|
| `app_windows.py` | Desktop GUI. Double-click via `run.bat` |
| `ui_kit.py` | Dark UI toolkit: cards, pills, toggles, metric tiles, motion |
| `humanizer_engine.py` | Offline engine, port of `AdvancedAIHumanizer` |
| `doc_tool.py` | Read PDF / Word / text, humanize per paragraph, write PDF |
| `humanize_cli.py` | Command-line version for batch/automation |
| `test_app.py` | Test suite - UI, engine and document pipeline |
| `run.bat` | Launcher - double-click to start the app |
| `build_exe.bat` | Builds the lean `dist\Humanizer.exe` (~11 MB) |
| `build_exe_neural.bat` | Builds `dist\Humanizer-Neural.exe` (~450 MB, models baked in) |
| `app.py`, `requirements.txt` | Your original Gradio version (kept untouched) |
| `requirements_optional.txt` | Optional neural upgrade (T5-small + MiniLM, like the HF Space) |

## Use (no install)

1. Install Python 3.9+ from https://www.python.org/downloads/ (tick **Add to PATH**)
2. Double-click **`run.bat`** - the app opens. No `pip install` needed.

To attach PDFs or Word files, add the two readers once:

```bat
pip install pypdf python-docx
```

Everything else - including saving a humanized PDF - needs nothing extra.

## Use as .exe (shareable, no Python needed)

Prebuilt binaries live in `dist\`:

| Binary | Size | What it does |
|---|---|---|
| `Humanizer.exe` | ~11 MB | Full offline engine. Neural toggle needs the pip install. |
| `Humanizer-Neural.exe` | ~450 MB | Same app with torch/transformers bundled. |

Rebuild either with `build_exe.bat` / `build_exe_neural.bat` (needs `pip install pyinstaller`).

## Command line

```bat
python humanize_cli.py input.txt -o output.txt --intensity standard --analysis
python humanize_cli.py --text "Furthermore, it is important to note that..." --intensity heavy
python humanize_cli.py input.txt --intensity heavy --neural
```

## Intensities

- **Light** - swaps AI-vocabulary only, keeps your sentence structure. Safest for meaning.
- **Standard** - vocabulary + sentence restructuring + natural flow. Recommended.
- **Heavy** - full restructuring: splits, merges, fragments, best-of-5 pick. For tough detectors.

## Tests

```bat
python test_app.py
```

Always proofread output - no humanizer preserves meaning 100%.
