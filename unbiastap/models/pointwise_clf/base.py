import logging
import pandas as pd

from unbiastap.models.base import BaseModelAdapter
from unbiastap.models.pointwise_clf.config import BasePointwiseClfConfig

logger = logging.getLogger(__name__)

class BasePointwiseClfAdapter(BaseModelAdapter):
    def __init__(self, config: BasePointwiseClfConfig):
        super().__init__()
        self.config = config
    
    def _ensure_dataframe(self, data):
        self._validate_data_format(data)
        columns = data.columns.tolist()

        missing = [
            col for col in [*self.config.features, self.config.label_column]
            if col not in columns
        ]
        
        if missing:
            raise ValueError(
                f"Missing columns in the input data: {', '.join(missing)}"
            )
    
    def _ensure_features(self, data):
        self._validate_data_format(data)

        missing = [
            col for col in self.config.features if col not in data.columns
        ]
        if missing:
            raise ValueError(
                f"Missing feature cols in the input data: {', '.join(missing)}"
            )

    def _ensure_fitted(self):
        if not self.is_fitted:
            logger.error(
                "%s.predict called before fit", type(self).__name__
            )
            raise RuntimeError(
                "Model must be fitted before making predictions."
            )
    
    def _validate_data_format(self, data):
        if not isinstance(data, pd.DataFrame):
            logger.error(
                "Expected input data to be a pandas DataFrame, but got %s",
                type(data).__name__
            )
            raise ValueError("Input data must be a pandas DataFrame.")