"""Launch this version's GR00T finetuner: python -m v3_bodymemory.train."""

import runpy


if __name__ == "__main__":
    runpy.run_module("v3_bodymemory.experiment.launch_finetune", run_name="__main__")
