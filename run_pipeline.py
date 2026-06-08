import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SELECTION_PATH = PROJECT_DIR / "selection.json"


def run_command(command, step_name):
    """Run a pipeline step and stop immediately if it fails."""
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{step_name} failed with exit code {exc.returncode}."
        ) from exc


def validate_selection():
    if not SELECTION_PATH.exists():
        raise RuntimeError("selection.json was not created. Selection step failed.")

    try:
        data = json.loads(SELECTION_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("selection.json could not be read.") from exc

    if not isinstance(data.get("selected_ids"), list):
        raise RuntimeError("selection.json does not contain a selected_ids list.")


def run_selection(input_video):
    print("\n[STEP 1] Player Selection\n")

    # Do not let a failed/cancelled selection reuse an older selection file.
    SELECTION_PATH.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        "--input",
        str(Path(input_video).resolve()),
    ]
    run_command(command, "Player selection")
    validate_selection()


def run_render(input_video, output_video):
    print("\n[STEP 2] Rendering Video\n")

    command = [
        sys.executable,
        str(PROJECT_DIR / "render_video.py"),
        "--input",
        str(Path(input_video).resolve()),
        "--output",
        str(Path(output_video).resolve()),
    ]
    run_command(command, "Video rendering")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    parser.add_argument("--skip-selection", action="store_true")
    args = parser.parse_args()

    if not args.skip_selection:
        run_selection(args.input)
    else:
        print("Skipping selection step")
        validate_selection()

    run_render(args.input, args.output)

    print("\nPipeline complete →", args.output)


if __name__ == "__main__":
    main()
