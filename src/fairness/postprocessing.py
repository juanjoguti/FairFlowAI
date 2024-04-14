import logging
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

logger = logging.getLogger(__name__)

class Postprocessing:
    """
    Postprocessing applies post-training bias mitigation techniques to model predictions.
    
    This class currently supports the Calibrated Equalized Odds post-processing technique from AIF360.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Postprocessing class with the specified mitigation technique and configuration.
        
        :param config: A dictionary containing configuration details, including the specified technique
                       for bias mitigation and the definitions for unprivileged and privileged groups.
        """
        self.technique = config['mitigation']['technique']
        self.unprivileged_groups = config['unprivileged_groups']
        self.privileged_groups = config['privileged_groups']
        self.techniques = {
            'CalibratedEqOdds': self.apply_calibrated_equalized_odds,
        }

    def apply(self, model, dataset):
        """
        Applies the specified postprocessing bias mitigation technique to model predictions.
        
        :param model: The trained machine learning model whose predictions need bias mitigation.
        :param dataset: The dataset for which predictions will be adjusted for fairness.
        :return: The dataset with postprocessed predictions.
        :raises ValueError: If an unsupported postprocessing technique is specified.
        """
        if self.technique in self.techniques:
            logger.info(f"Applying postprocessing technique: {self.technique}")
            return self.techniques[self.technique](model, dataset)
        else:
            logger.error("Unsupported postprocessing technique")
            raise ValueError("Unsupported postprocessing technique")

    def apply_calibrated_equalized_odds(self, model, dataset):
        """
        Applies Calibrated Equal Odds post-processing from AIF360 to adjust predictions of a trained model.
        
        :param model: The trained machine learning model.
        :param dataset: The dataset as a BinaryLabelDataset from AIF360 used for predictions.
        :return: A dataset with labels adjusted to achieve calibrated equalized odds.
        """
        logger.info("Applying Calibrated Equal Odds post-processing.")
        
        y_pred = model.predict(dataset.features)
        pred_dataset = dataset.copy()
        pred_dataset.labels = y_pred

        ceo = CalibratedEqOddsPostprocessing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            seed=None,
        )
        ceo = ceo.fit(dataset, pred_dataset)

        postprocessed_pred = ceo.predict(pred_dataset).labels
        logger.info("Calibrated Equal Odds post-processing applied.")

        return postprocessed_pred
