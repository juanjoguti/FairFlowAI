import logging
from aif360.algorithms.preprocessing import Reweighing

logger = logging.getLogger(__name__)

class Preprocessing:
    """
    Preprocessing manages the application of preprocessing techniques to mitigate bias before
    training machine learning models.
    
    It currently supports the Reweighing method from AIF360, with the potential to add more techniques.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Preprocessing class with the specified mitigation technique.
        
        :param config: A dictionary containing configuration details, specifically the technique
                       for bias mitigation.
        """
        self.technique = config['mitigation']['technique']
        self.unprivileged_groups = config['unprivileged_groups']
        self.privileged_groups = config['privileged_groups']
        self.techniques = {
            'Reweighing': self.apply_reweighing,
        }

    def apply(self, dataset=None):
        """
        Applies the specified preprocessing bias mitigation technique.
        
        :param dataset: The dataset to which the bias mitigation technique will be applied.
        :return: The dataset after applying the specified bias mitigation technique.
        :raises ValueError: If an unsupported preprocessing technique is specified.
        """
        if self.technique in self.techniques:
            logger.info(f"Applying preprocessing technique: {self.technique}")
            return self.techniques[self.technique](dataset)
        else:
            logger.error("Unsupported preprocessing technique")
            raise ValueError("Unsupported preprocessing technique")

    def apply_reweighing(self, dataset):
        """
        Applies the Reweighing preprocessing technique from AIF360 to mitigate bias in the dataset.
        
        :param dataset: AIF360's BinaryLabelDataset to be mitigated.
        :return: The mitigated dataset with adjusted instance weights.
        """
        logger.info("Applying Reweighing for bias mitigation.")
        reweigh = Reweighing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
        )
        mitigated_dataset = reweigh.fit_transform(dataset)
        logger.info("Reweighing applied successfully.")
        return mitigated_dataset
