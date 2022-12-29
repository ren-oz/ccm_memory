import numpy as np
from abstract import AbstractMemory, AbstractMemoryItem
from abc import ABC, abstractmethod


class Vector(np.ndarray, AbstractMemoryItem): ...


class Activation(Vector): ...


class MemoryPair:
    def __init__(self, memory_model: type, *args, **kwargs) -> None:
        self._val = (memory_model(*args, **kwargs), memory_model(*args, **kwargs))
    
    def __getitem__(self, i):
        return self._val[i]

    def __str__(self):
        return f"({str(self._val[0])}, {str(self._val[1])})"


class AbstractMatrixMemory(AbstractMemory, ABC):
    @abstractmethod
    def activation(self, probe: Vector) -> Activation: ...
    
    @abstractmethod
    def apply_activation(self, activation: Activation) -> Vector: ...

    def __init__(self, matrix:np.ndarray=None):
        self._memory = matrix
    
    def add(self, item: Vector) -> None:
        v = np.array(item)
        if self._memory is None:
            self._memory = v.reshape((1, v.size))
        else:
            self._memory = np.concatenate((self._memory, v.reshape((1, v.size))))
    
    def retrieve(self, probe: Vector) -> Vector:
        if self._memory is None:
            return Vector(0)
        return self.apply_activation(self.activation(probe)).view(Vector)
        
    def delete(self, index:int) -> None:
        n = self._memory.shape[0]
        i = np.linspace(0, n, num=n, endpoint=False)
        self._memory = self._memory[np.r_[i[:index], i[index+1:]]]
    
    def __str__(self):
        return str(self.memory)

    def __len__(self):
        return self._memory.shape[0] if self._memory is not None else 0
    
    @property
    def memory(self):
        return self._memory if self._memory is not None else Vector(0)
