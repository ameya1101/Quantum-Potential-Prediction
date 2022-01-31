from abc import ABC, abstractmethod

class BaseSystem(ABC):

    def __init__(self, dim, domain) -> None:
        self._dim = dim
        assert len(domain) == self._dim, f"System with dimension {self._dim} cannot have {len(domain)} domains."
        self._domains = domain
        super().__init__()

    @abstractmethod
    def wavefunction(self, x):
        pass