import logging
import yaml
from sklearn.base import is_classifier, is_regressor

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.fairness.mitigator import BiasMitigator
from src.evaluation.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class ModelPipeline:
    """
    ModelPipeline coordinates the various stages in the data loading, preprocessing,
    model training, bias mitigation, and evaluation process.
    """

    def __init__(self):
        """
        Initializes the pipeline, creating instances of DataLoader and DataPreprocessor.
        """
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()

    def run(self, config_path, model):
        """
        Executes the pipeline using the configuration specified and the given model.

        :param config_path: Path to the YAML configuration file.
        :param model: A scikit-learn compatible model that will be trained and evaluated.
        :return: Tuple of (trained model, performance metrics, fairness metrics).
        """
        self.validate_input(model)
        self.read_config(config_path)

        dataset = self.data_loader.load_dataset(self.config['dataset_name'])
        train, test = self.data_preprocessor.split_dataset(dataset)
        train, test = self.preprocess_data(train, test)
        model = self.fit(model, train)
        y_pred = self.predict(model, test)
        performance_metrics, fairness_metrics = self.evaluate(test, y_pred)

        return model, performance_metrics, fairness_metrics

    def read_config(self, config_path):
        """
        Reads the model configuration from a YAML file and initializes the BiasMitigator.
        
        :param config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.bias_mitigator = BiasMitigator(self.config)
        logger.info("Configuration loaded and bias mitigator initialized.")

    def validate_input(self, model):
        """
        Validates the input model to ensure it is a classifier or regressor.

        :param model: The model to validate.
        :raises ValueError: If the model is not a classifier or regressor.
        """
        if not (is_classifier(model) or is_regressor(model)):
            logger.error("Invalid model input: Model is not a valid sklearn classifier or regressor.")
            raise ValueError("The provided model is not a valid sklearn classifier or regressor.")

    def preprocess_data(self, train, test):
        """
        Applies preprocessing steps to the data, including bias mitigation if specified.
        
        :param train: The training dataset.
        :param test: The test dataset.
        :return: Tuple of (train, test) datasets after preprocessing.
        """
        if self.config.get('mitigation_strategy') == 'Preprocessing':
            logger.info("Applying preprocessing bias mitigation.")
            train = self.bias_mitigator.apply(dataset=train)
        
        if 'scaler' in self.config:
            logger.info("Applying data scaling.")
            train, test = self.data_preprocessor.apply_scaling(
                train, test, self.config['scaler']
            )
        
        return train, test

    def fit(self, model, train):
        """
        Trains the model using the training data, applying in-processing bias mitigation if specified.
        
        :param model: The model to be trained.
        :param train: The training dataset.
        :return: The trained model.
        """
        if self.config.get('mitigation_strategy') == 'Inprocessing':
            logger.info("Applying inprocessing bias mitigation.")
            model = self.bias_mitigator.apply(model=model, dataset=train)
        else:
            logger.info("Fitting model to training data.")
            model.fit(train.features, train.labels.ravel())
        
        return model

    def predict(self, model, test):
        """
        Generates predictions using the trained model, applying post-processing bias mitigation if specified.
        
        :param model: The trained model.
        :param test: The test dataset.
        :return: The model's predictions.
        """
        if self.config.get('mitigation_strategy') == 'Postprocessing':
            logger.info("Applying postprocessing bias mitigation.")
            y_pred = self.bias_mitigator.apply(model=model, dataset=test)
        else:
            logger.info("Generating model predictions.")
            y_pred = model.predict(test.features)
        
        return y_pred

    def evaluate(self, test, y_pred):
        """
        Evaluates the performance and fairness of the model's predictions.
        
        :param test: The test dataset.
        :param y_pred: The: Predictions made by the model.
        return: A tuple containing dictionaries of performance and fairness metrics.
        """
        model_evaluator = ModelEvaluator(self.config)

        logger.info("Evaluating model performance.")
        performance_metrics = model_evaluator.evaluate_performance(
            test.labels, y_pred
        )
        
        logger.info("Evaluating model fairness.")
        fairness_metrics = model_evaluator.evaluate_fairness(test, y_pred)
        
        return performance_metrics, fairness_metrics
