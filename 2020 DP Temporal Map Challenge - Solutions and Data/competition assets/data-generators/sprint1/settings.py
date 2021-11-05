import json
from pathlib import Path

from pydantic import BaseSettings

LIBRARY_ROOT = Path(__file__).parents[1]
PROJECT_DIR = LIBRARY_ROOT.parent


class Settings(BaseSettings):
    RANDOM_SEED: int = 42
    PROJECT_DIR: Path = PROJECT_DIR
    DATA_DIR: Path = PROJECT_DIR / "data"
    CONFIG_FILE: Path = PROJECT_DIR / "../deid2-runtime/data/parameters.json"

    def get_parameters(self) -> dict:
        with self.CONFIG_FILE.open("r") as fp:
            return json.load(fp)


settings = Settings()
