from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClipSelectorConfig(BaseModel):
    n_grid_points: int = Field(
        100,
        description=(
            "Number of candidate clip values to evaluate across the weight "
            "distribution. Candidates are drawn in percentile space."
        ),
    )
    tau: float = Field(
        0.3,
        description=(
            "Minimum acceptable ESS/n ratio. "
            "The selector returns the smallest clip value c "
            "such that ESS(c)/n >= tau. A value of 0.3 means at "
            "least 30% of statistical power must be retained after clipping."
        ),
    )

    @field_validator("tau", mode="before")
    @classmethod
    def validate_tau(cls, value: float) -> float:
        if not (0 < value <= 1):
            raise ValueError("tau must be in (0, 1].")
        return value


class ShrinkageSelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n0_min: float = Field(
        10.0,
        description=(
            "Minimum prior strength to search. At n0=10, the empirical estimate "
            "dominates even at positions with very few sessions."
        ),
    )
    n0_max: float = Field(
        100_000.0,
        description=(
            "Maximum prior strength to search. At n0=100k, the blend falls back "
            "almost entirely to the parametric model at all positions."
        ),
    )
    n_grid_points: int = Field(
        50,
        description="Number of log-spaced candidates between n0_min and n0_max.",
    )
    random_seed: int = Field(
        42,
        description=(
            "Seed for the fold simulation. Controls how sessions are split "
            "between fold A (training) and fold B (holdout) at each position."
        ),
    )

    @model_validator(mode="after")
    def validate_n0_range(self) -> "ShrinkageSelectorConfig":
        if self.n0_min >= self.n0_max:
            raise ValueError("n0_min must be strictly less than n0_max.")
        return self
