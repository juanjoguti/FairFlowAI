import logging
from fairlearn.reductions import DemographicParity, GridSearch

logger = logging.getLogger(__name__)

class Inprocessing:
    """
    Inprocessing handles the application of in-processing techniques to mitigate bias during
    the training of machine learning models.
    
    Currently supports the ExponentiatedGradient method from Fairlearn, with the possibility
    of adding more techniques.
    """

    def __init__(self, config) -> None:
        """
        Initializes the Inprocessing class with the specified mitigation technique.
        
        :param config: A dictionary containing configuration details, specifically the 
                       technique for bias mitigation.
        """
        self.technique = config['mitigation']['technique']
        self.techniques = {
            'DemographicParity': self.apply_demographic_parity_grid_search,
        }
    
    def apply(self, model=None, dataset=None):
        """
        Applies the specified in-processing bias mitigation technique.
        
        :param model: The machine learning model that requires bias mitigation.
        :param dataset: The dataset on which the model will be trained.
        :return: A mitigator object that has been fit to the dataset.
        :raises ValueError: If an unsupported in-processing technique is specified.
        """
        logger.info(f"Applying inprocessing technique: {self.technique}")
        if self.technique in self.techniques:
            return self.techniques[self.technique](model, dataset)
        else:
            logger.error("Unsupported inprocessing technique")
            raise ValueError("Unsupported inprocessing technique")

    def apply_demographic_parity_grid_search(self, model, dataset):
        """
        Applies GridSearch with a Demographic Parity constraint.
        
        :param model: The machine learning model that requires bias mitigation.
        :param dataset: The dataset on which the model will be trained.
        :return: A mitigator object that has been fit to the dataset.
        :raises ValueError: If an unsupported in-processing technique is specified.
        """
        X, y = dataset.features, dataset.labels.ravel()
        sensitive_features = dataset.protected_attributes[:,0]
        
        logger.info("Fitting GridSearch with DemographicParity constraint")
        mitigator = GridSearch(
            model,
            constraints=DemographicParity(),
            grid_size=50,
        )
        
        mitigator.fit(X, y, sensitive_features=sensitive_features)
        logger.info("GridSearch with DemographicParity constraint mitigation applied")

        return mitigator
