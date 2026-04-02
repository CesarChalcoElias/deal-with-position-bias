import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from unbiastap.models.pointwise_clf.base import BasePointwiseClfAdapter
from unbiastap.models.pointwise_clf.config import IPSXGBoostConfig

logger = logging.getLogger(__name__)

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

        n_clipped = int((weights > clip_value).sum())
        if n_clipped > 0:
            logger.warning(
                "Clipping %d IPS weights (%.1f%%) above the %.1f percentile "
                "(clip value: %.4f). Large clipping ratios may indicate poor "
                "propensity score calibration.",
                n_clipped,
                100 * n_clipped / len(weights),
                self.config.weight_clip_percentile,
                clip_value,
            )

        weights = np.clip(weights, a_min=None, a_max=clip_value)
        logger.debug(
            "IPS weights — min: %.4f, mean: %.4f, max: %.4f",
            weights.min(), weights.mean(), weights.max(),
        )
        return weights

    def fit(self, data: pd.DataFrame):
        logger.info(
            "Fitting %s on %d samples with %d features",
            type(self).__name__, len(data), len(self.config.features),
        )
        self._ensure_dataframe(data)
        sample_weights = self._compute_ips_weights(data)

        X = data.loc[:, self.config.features]
        y = data.loc[:, self.config.label_column]

        self.model_.fit(X, y, sample_weight=sample_weights)
        self.is_fitted = True
        logger.info("%s fitted successfully", type(self).__name__)
        return self

    def predict(self, data: pd.DataFrame):
        self._ensure_fitted()
        self._ensure_features(data)
        logger.debug("Running predict on %d samples", len(data))
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
        logger.info("Model saved to %s with prefix '%s'", base, prefix)

    @classmethod
    def load(cls, path: str, prefix: str = "ips_xgboost"):
        base = Path(path)
        logger.info("Loading model from %s with prefix '%s'", base, prefix)
        with open(base / f"{prefix}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(base / f"{prefix}_config.json", "r") as f:
            config_dict = json.load(f)
        config = IPSXGBoostConfig.model_validate(config_dict)
        adapter = cls(config)
        adapter.model_ = model
        adapter.is_fitted = True
        logger.info("%s loaded successfully", cls.__name__)
        return adapter

    def get_shap_booster(self):
        self._ensure_fitted()   
        return self.model_.get_booster()