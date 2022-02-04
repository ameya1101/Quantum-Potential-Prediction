from abc import ABC, abstractmethod

class BaseSystem(ABC):
    '''
    Base class for a quantum mechanical system.
    '''
    def __init__(self, dim, domain) -> None:
        self.dim = dim
        assert len(domain) == self.dim, f"System with dimension {self.dim} cannot have {len(domain)} domains."
        self.domains = domain
        super().__init__()

    @abstractmethod
    def potential(self, x):
        ''' 
        Implement the potential energy function for the system
        '''
        pass

    @abstractmethod
    def wavefunction(self, x):
        '''
        Implement the wavefunction for the system
        '''
        pass
    