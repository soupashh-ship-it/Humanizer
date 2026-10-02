"""Command-line humanizer for Windows.

Usage:
  python humanize_cli.py input.txt -o output.txt --intensity standard
  python humanize_cli.py --text "Furthermore, it is important to note..." --intensity heavy
  echo some text | python humanize_cli.py --intensity light
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from humanizer_engine import AdvancedAIHumanizer


def main():
    p = argparse.ArgumentParser(description="Humanizer - offline AI text humanizer")
    p.add_argument("input", nargs="?", help="Input .txt file (omit to use --text or stdin)")
    p.add_argument("-o", "--output", help="Output .txt file (default: print to console)")
    p.add_argument("--text", help="Direct text input")
    p.add_argument("--intensity", default="standard",
                   choices=["light", "standard", "heavy"],
                   help="Humanization intensity (default: standard)")
    p.add_argument("--analysis", action="store_true",
                   help="Also print detection analysis")
    p.add_argument("--neural", action="store_true",
                   help="Use T5-small + MiniLM models (same as the HuggingFace "
                        "Space; slower, stronger; needs requirements_optional.txt)")
    args = p.parse_args()

    if args.text:
        text = args.text
    elif args.input:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        p.print_help()
        print("\nError: provide a file, --text, or pipe text via stdin.")
        sys.exit(1)

    h = AdvancedAIHumanizer()
    if args.neural:
        ok = h.enable_neural_models()
        print(f"Neural models: {h.neural_status()}", file=sys.stderr)
        if not ok:
            print("Continuing with offline rule-based engine.", file=sys.stderr)
    result = h.humanize_text(text, args.intensity)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Saved to {args.output}")
    else:
        print(result)

    if args.analysis:
        print("\n" + "=" * 60)
        print(h.get_detailed_analysis(result))
        print(f"\nWords changed: {h.change_percent(text, result):.0f}% "
              f"(site targets: Light ~70% / Standard ~85% / Heavy ~95%)")
        print(f"Engine: {h.neural_status()}")


if __name__ == "__main__":
    main()
