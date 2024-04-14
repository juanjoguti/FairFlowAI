import logging
from src.fairness.inprocessing import Inprocessing
from src.fairness.postprocessing import Postprocessing
from src.fairness.preprocessing import Preprocessing

logger = logging.getLogger(__name__)

class BiasMitigator:
    """
    BiasMitigator orchestrates the application of different bias mitigation strategies, 
    including preprocessing, inprocessing, and postprocessing techniques.
    
    It allows for a flexible application of the specified strategy on the given model and dataset.
    """

    def __init__(self, config):
        """
        Initializes the BiasMitigator with the specified mitigation strategy and associated configuration.
        
        :param config: A dictionary containing the mitigation strategy and configuration details 
                       for unprivileged and privileged groups.
        """
        self.strategy = config['mitigation']['strategy']
        self.unprivileged_groups = config['unprivileged_groups']
        self.privileged_groups = config['privileged_groups']
        
        preprocessing = Preprocessing(config)
        inprocessing = Inprocessing(config)
        postprocessing = Postprocessing(config)
        
        self.strategies = {
            'Preprocessing': preprocessing.apply,
            'Inprocessing': inprocessing.apply,
            'Postprocessing': postprocessing.apply
        }
        logger.info(f"Bias mitigation strategy set to: {self.strategy}")

    def apply(self, model=None, dataset=None):
        """
        Applies the configured bias mitigation strategy to the given model and dataset.
        
        :param model: The machine learning model to which the bias mitigation technique will be applied.
        :param dataset: The dataset that may be used during the mitigation process.
        :return: The result of the bias mitigation technique, which may be a modified dataset or model.
        :raises ValueError: If an unsupported mitigation strategy is specified.
        """
        logger.info(f"Applying {self.strategy} bias mitigation.")
        if self.strategy in self.strategies:
            return self.strategies[self.strategy](model, dataset)
        else:
            logger.error("Unsupported mitigation strategy")
            raise ValueError("Unsupported mitigation strategy")
