import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from unbiastap.models.pointwise_clf.base import BasePointwiseClfAdapter
from unbiastap.models.pointwise_clf.config import IPSXGBoostConfig

class IPSXGBoostAdapter(BasePointwiseClfAdapter):
    def __init__(self, config: IPSXGBoostConfig):
        super().__init__(config)
        self.model_ = XGBClassifier(**config.hyperparams.model_dump())

    def _compute_ips_weights(self, data: pd.DataFrame):
        if self.config.exposure_propensity_column not in data.columns:
            raise ValueError(
                f"Exposure propensity column "
                f"'{self.config.exposure_propensity_column}' is missing from "
                f"the input data."
            )
        
        propensity = data[self.config.exposure_propensity_column].to_numpy()
        weights = 1.0 / propensity
        clip_value = np.percentile(weights, self.config.weight_clip_percentile)
        weights = np.clip(weights, a_min=None, a_max=clip_value)
        return weights

    def fit(self, data: pd.DataFrame):
        self._ensure_dataframe(data)
        sample_weights = self._compute_ips_weights(data)

        X = data.loc[:, self.config.features]
        y = data.loc[:, self.config.label_column]

        self.model_.fit(X, y, sample_weight=sample_weights)
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame):
        self._ensure_fitted()
        self._ensure_features(data)
        X = data.loc[:, self.config.features]
        return self.model_.predict_proba(X)[:, 1]
    
    def save(self, path: str, prefix: str = "ips_xgboost"):
        self._ensure_fitted()
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        with open(base / f"{prefix}_model.pkl", "wb") as f:
            pickle.dump(self.model_, f)
        with open(base / f"{prefix}_config.json", "w") as f:
            json.dump(self.config.model_dump(), f)
        
    @classmethod
    def load(cls, path: str, prefix: str = "ips_xgboost"):
        base = Path(path)
        with open(base / f"{prefix}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(base / f"{prefix}_config.json", "r") as f:
            config_dict = json.load(f)
        config = IPSXGBoostConfig.model_validate(config_dict)
        adapter = cls(config)
        adapter.model_ = model
        adapter.is_fitted = True
        return adapter

    def get_shap_booster(self):
        self._ensure_fitted()   
        return self.model_.get_booster()