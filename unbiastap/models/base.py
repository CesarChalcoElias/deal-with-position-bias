from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, data): ...

    @abstractmethod
    def predict(self, data): ...

    @abstractmethod
    def save(self, path): ...

    @classmethod
    @abstractmethod
    def load(cls, path): ...

    @abstractmethod
    def get_shap_booster(self): ...
