from abc import ABC, abstractmethod

class BaseSystem(ABC):

    def __init__(self, dim, domain) -> None:
        self.dim = dim
        assert len(domain) == self.dim, f"System with dimension {self.dim} cannot have {len(domain)} domains."
        self.domains = domain
        super().__init__()

    @abstractmethod
    def potential(self, x):
        pass

    @abstractmethod
    def wavefunction(self, x):
        pass