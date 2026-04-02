from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

class PointwiseClfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_column: str = Field(
        ...,
        description=(
            "The name of the column in the dataset that contains the "
            "labels for classification."
        ),
    )
    features: list[str] = Field(
        default_factory=list,
        description=(
            "A list of column names in the dataset that should be used as "
            "features for training the classification model."
        ),
    )

class IPSXGBoostHyperparams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_depth: int = 8
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

class IPSXGBoostConfig(PointwiseClfConfig):
    model_config = ConfigDict(extra="forbid")
    exposure_propensity_column: str = Field(
        "exposure_propensity",
        description=(
            "The name of the column in the dataset that contains the "
            "propensity scores for exposure. These scores are used to "
            "compute the importance weights for the IPS estimator."
        ),
    )
    hyperparams: IPSXGBoostHyperparams = Field(
        default_factory=IPSXGBoostHyperparams,
        description="Hyperparameters for the XGBoost model.",
    )
    weight_clip_percentile: float = Field(
        99.0,
        description=(
            "The percentile for clipping the importance weights. This is used "
            "to prevent excessively large weights that can lead to "
            "instability in training. For example, if set to 99.0, weights "
            "above the 99th percentile will be clipped to the value at the "
            "99th percentile."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("Input must be a dictionary.")
        
        hyperparams_keys = set(IPSXGBoostHyperparams.model_fields.keys())

        flat_hp = {k: v for k, v in value.items() if k in hyperparams_keys}

        if flat_hp:                                                                                                              
            value = {
                **value,
                "hyperparams": {
                    **value.get("hyperparams", {}), **flat_hp
                }
            }
            value = {
                k: v for k, v in value.items() if k not in hyperparams_keys
            }
                                                                                                                                
        return value 

