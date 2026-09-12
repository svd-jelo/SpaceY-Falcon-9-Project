import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV

from spacey_falcon_9_project.config import models_dir, processed_dir
from spacey_falcon_9_project.utils.modeling import ArtifactSaverLoader


class BatchInference(object):
    def __init__(self, input_path, output_path, model_path, model_name):
        self.input_path = self._load_path(input_path)
        self.output_path = self._load_path(output_path)
        self.model_path = self._load_path(model_path)
        self.model_name = model_name

    @staticmethod
    def _load_path(path: str|Path) -> Path:
        if isinstance(path, str):
            pathlib_path = Path(path)
        elif isinstance(path, Path):
            pathlib_path = path
        else:
            raise ValueError("Input path must be a string or a Path object")
        return pathlib_path

    def load_data(self):
        if self.input_path.is_file():
            data = pd.read_csv(self.input_path)
            if 'class' in data.columns:
                data = data.drop(columns='class')
        else:
            raise FileNotFoundError("Input path does not exist")
        return data

    def load_model(self) -> TunedThresholdClassifierCV:
        if self.model_path.joinpath(self.model_name).is_file():
            artifact_interface = ArtifactSaverLoader(models_dir=self.model_path)
            model = artifact_interface.load_artifact(artifact_name=self.model_name)
        else:
            raise FileNotFoundError("Artifact does not exist")
        return model

    def write_to_predictions(self, df: pd.DataFrame) -> bool:
        df.to_csv(self.output_path, index=False)
        return self.output_path.is_file()

    def predict(self):
        logger = logging.getLogger('batch_predict')

        logger.info(f"Loading data from {self.input_path}")
        data_df = self.load_data()

        logger.info(f"Loading model from {self.model_path} - {self.model_name}")
        model = self.load_model()

        logger.info(f"Predicting on {data_df.shape[0]} samples")
        data_df["predicted_price"] = model.predict(data_df)

        logger.info(f"Saving predictions to {self.output_path}")
        self.write_to_predictions(data_df)

        logger.info(f"Finished predicting on {data_df.shape[0]} samples")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Run batch inference.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(processed_dir.joinpath('sample_input.csv')),
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(processed_dir.joinpath('sample_output.csv')),
        help="Path to output data CSV"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(models_dir),
        help="Directory where the model artifact is stored"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Filename of the model artifact (e.g., svc-2026-09-11T14-30-00Z.joblib)"
    )

    args = parser.parse_args()

    predictor = BatchInference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_dir,
        model_name=args.model_name
    )

    predictor.predict()