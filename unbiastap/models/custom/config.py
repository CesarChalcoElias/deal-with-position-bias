from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClipSelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
