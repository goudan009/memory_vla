"""Launch this version's GR00T finetuner: python -m v1_groot_memoryvla.train."""

import runpy


if __name__ == "__main__":
    runpy.run_module("v1_groot_memoryvla.experiment.launch_finetune", run_name="__main__")
