"""Launch the pinned, unmodified N1.6 reference finetuner."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    reference = Path(__file__).resolve().parents[2] / "src/references_code/Isaac-GR00T"
    sys.path.insert(0, str(reference))
    runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")
