import logging

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas
)

logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader is responsible for loading various preprocessed datasets.
    
    It uses a mapping of dataset names to their respective loading functions
    provided by the AIF360 toolkit.
    """

    def __init__(self):
        """
        Initializes a new instance of the DataLoader class.
        """
        self.dataset_loaders = {
            'adult': load_preproc_data_adult,
            'compas': load_preproc_data_compas,
            'german': load_preproc_data_german,
        }
    
    def load_dataset(self, dataset_name):
        """
        Loads the specified dataset using a mapping to the appropriate
        AIF360 loading function.
        
        :param dataset_name: Name of the dataset to load.
        :return: An instance of the specified dataset.
        :raises ValueError: If the specified dataset is not supported.
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            return self.dataset_loaders[dataset_name]()
        except KeyError:
            logger.error(f"Dataset {dataset_name} is not supported")
            raise ValueError(f"Dataset {dataset_name} is not supported")
