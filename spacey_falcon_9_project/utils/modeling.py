from datetime import datetime
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import TunedThresholdClassifierCV

from spacey_falcon_9_project.config import root


class ArtifactSaverLoader(object):
    def __init__(self, models_dir: str | Path = Path(root / "models")):
        self.models_dir = models_dir

    def save_artifact(self, artifact: BaseEstimator, artifact_name: None | str = None) -> bool:
        today = datetime.today().strftime("%Y-%m-%dT%H-%M-%SZ")
        artifact_unique_name = (
            f"{artifact_name}-{today}.joblib" if artifact_name else f"artifact-{today}.joblib"
        )
        artifact_path = Path(self.models_dir) / artifact_unique_name
        joblib.dump(artifact, artifact_path)
        return artifact_path.is_file()

    def load_artifact(self, artifact_name: str) -> TunedThresholdClassifierCV:
        artifact_path = Path(self.models_dir) / artifact_name
        if artifact_path.is_file():
            model = joblib.load(artifact_path)
        else:
            raise FileNotFoundError(f"Artifact {artifact_name} not found")
        return model
