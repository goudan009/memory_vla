"""Load a three-task config and call NVIDIA's unchanged training entrypoint."""
import json
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run


if __name__ == "__main__":
    config_path = Path(__file__).with_name("train.json")
    config = get_default_config().load_dict(json.loads(config_path.read_text()))
    run(config)
