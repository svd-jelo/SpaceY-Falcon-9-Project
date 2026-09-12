import logging
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from spacey_falcon_9_project.config import models_dir, processed_dir, random_state, training_set
from spacey_falcon_9_project.features import (
    build_features,
    load_test_set,
    load_training_set,
    load_whole_dataset,
)
from spacey_falcon_9_project.utils.metrics import gain_score
from spacey_falcon_9_project.utils.modeling import ArtifactSaverLoader


class ModelTrainer(object):
    def __init__(self,
                 input_filepath: str|Path = processed_dir / training_set,
                 output_filepath: str|Path = models_dir,
                 model_name: None|str = None):

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.model_name = model_name

    @staticmethod
    def fit_model(X: pd.DataFrame, y: pd.Series) -> TunedThresholdClassifierCV:
        transformer_step = ('transformer', build_features())
        classifier_step = ('classifier', SVC(class_weight='balanced', random_state=random_state))
        svc_param_grid = [
            {
                "classifier__kernel": ["linear"],
                "classifier__C": loguniform(1e1, 1e2),
            },
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": loguniform(5 * 1e2, 5 * 1e3),
                "classifier__gamma": loguniform(1e-3, 1e-2),
            },
            {
                "classifier__kernel": ["poly"],
                "classifier__C": loguniform(1e0, 1e1),
                "classifier__gamma": loguniform(1e-3, 1e-1),
                "classifier__degree": [2, 3, 4],
            },
        ]
        svc_model = RandomizedSearchCV(
            Pipeline([transformer_step, classifier_step]),
            svc_param_grid,
            scoring = 'average_precision',
            cv = 5,
            random_state=random_state,
            n_iter = 100,
            n_jobs = -1,
        )
        svc_model.fit(X, y)

        weights = {'weight_fp': 10, 'weight_fn': 1}
        scorer = make_scorer(gain_score, **weights)
        svc_model_tuned = TunedThresholdClassifierCV(svc_model.best_estimator_,scoring=scorer, cv=5, random_state=random_state)
        svc_model_tuned.fit(X, y)
        return svc_model_tuned

    @staticmethod
    def evaluate_model(model: TunedThresholdClassifierCV,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        weights = {'weight_fp': 10, 'weight_fn': 1}
        scorer = make_scorer(gain_score, **weights)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        evaluation = {
            'train_score': {
                'gain_score': scorer(model, X_train, y_train),
                'precision_score': precision_score(y_train, y_train_pred),
                'recall_score': recall_score(y_train, y_train_pred),
            },
            'test_score': {
                'gain_score': scorer(model, X_test, y_test),
                'precision_score': precision_score(y_test, y_test_pred),
                'recall_score': recall_score(y_test, y_test_pred),
            }
        }
        return evaluation

    def train(self) -> None:
        logger = logging.getLogger('train_model')

        logger.info(f'Loading training data from {self.input_filepath}')
        X_train, y_train = load_training_set()

        logger.info(f'Loading testing data from {self.input_filepath}')
        X_test, y_test = load_test_set()

        logger.info('Fitting model to training data')
        trained_model = self.fit_model(X_train, y_train)

        logger.info('Evaluating model')
        evaluation = self.evaluate_model(trained_model, X_train, y_train, X_test, y_test)

        logger.info(f'Training set:'
                    f'\n gain_score: {evaluation['train_score']['gain_score']}'
                    f'\n precision: {evaluation['train_score']['precision_score']}'
                    f'\n recall_score: {evaluation['train_score']['recall_score']}')

        logger.info(f'Testing set:'
                    f'\n gain_score: {evaluation['test_score']['gain_score']}'
                    f'\n precision_score: {evaluation['test_score']['precision_score']}'
                    f'\n recall_score: {evaluation['test_score']['recall_score']}')

        logger.info('Save artifacts')
        artifact_interface = ArtifactSaverLoader(models_dir=self.output_filepath)
        artifact_interface.save_artifact(trained_model, artifact_name=self.model_name)

        logger.info('Done!')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    model_name = 'svc'
    trainer = ModelTrainer(model_name=model_name)
    trainer.train()




