import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    DataPreprocessor prepares the dataset for the machine learning model.
    
    It provides functionality to split the dataset into training and testing sets
    and apply feature scaling.
    """
    def __init__(self):
        """
        Initializes a new instance of the DataPreprocessor class.
        """
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler()
        }

    def split_dataset(self, dataset):
        """
        Splits the dataset into training and testing sets with a default 70-30 split.
        
        :param dataset: A dataset instance that supports the split method.
        :return: A tuple (train, test) containing the split datasets.
        """
        train, test = dataset.split([0.7], shuffle=True)
        logger.info("Dataset split into train and test sets.")
        return train, test
    
    def apply_scaling(self, train, test, scaler_name):
        """
        Applies feature scaling to the training and testing sets using the specified scaler.
        
        :param train: The training dataset.
        :param test: The testing dataset.
        :param scaler_name: A string identifier for the scaler to be used.
        :return: The training and testing sets after scaling has been applied.
        :raises ValueError: If the specified scaler is not supported.
        """
        if scaler_name not in self.scalers:
            logger.error(f"Scaler {scaler_name} is not supported.")
            raise ValueError(f"Scaler {scaler_name} is not supported.")
        
        scaler = self.scalers[scaler_name]
        logger.info(f"Applying {scaler_name} to the features.")
        train.features = scaler.fit_transform(train.features)
        test.features = scaler.transform(test.features)
        logger.info("Feature scaling applied.")

        return train, test
