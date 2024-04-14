import logging
from aif360.metrics import ClassificationMetric
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    ModelEvaluator assesses the performance and fairness of machine learning models.
    
    It utilizes standard performance metrics from scikit-learn and fairness
    metrics from the AIF360 library.
    """

    def __init__(self, config):
        """
        Initializes the ModelEvaluator with configuration for fairness evaluation.
        
        :param config: A dictionary containing 'unprivileged_groups' and 'privileged_groups' 
                       for fairness metrics.
        """
        self.unprivileged_groups = config['unprivileged_groups']
        self.privileged_groups = config['privileged_groups']

    def evaluate_performance(self, y_test, y_pred, y_pred_proba=None):
        """
        Evaluates the performance of the model using standard metrics.
        
        :param y_test: The ground truth labels.
        :param y_pred: The predicted labels by the model.
        :param y_pred_proba: The predicted probabilities by the model (optional).
        :return: A dictionary of performance metrics.
        """
        # Compute and return the standard performance metrics.
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, average='binary'),
            "Precision": precision_score(y_test, y_pred, average='binary'),
            "Recall": recall_score(y_test, y_pred, average='binary')
        }
        if y_pred_proba is not None:
            metrics["AUC-ROC"] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("Performance metrics evaluated.")
        return metrics
    
    def evaluate_fairness(self, dataset, y_pred):
        """
        Evaluates the fairness of the model using AIF360 metrics.
        
        :param dataset: The original dataset used for predictions.
        :param y_pred: The predicted labels by the model.
        :return: A dictionary of fairness metrics.
        """
        y_pred_reshaped = y_pred.reshape(-1, 1)
        predictions_dataset = dataset.copy()
        predictions_dataset.labels = y_pred_reshaped
        
        metric = ClassificationMetric(
            dataset,
            predictions_dataset,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
        )
        fairness_metrics = {
            "Equal Opportunity Difference": metric.equal_opportunity_difference(),
            "Statistical Parity Difference": metric.statistical_parity_difference(),
            "Average Odds Difference": metric.average_odds_difference(),
            "Disparate Impact": metric.disparate_impact()
        }
        
        logger.info("Fairness metrics evaluated.")
        return fairness_metrics
