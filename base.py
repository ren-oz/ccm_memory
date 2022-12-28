import numpy as np
from abstract import AbstractMemory, AbstractMemoryItem
from abc import ABC, abstractmethod


class Vector(np.ndarray, AbstractMemoryItem): ...


class Activation(Vector): ...


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
        self._memory = self._memory[[i for i in range(self._memory.shape[0]) if i != index]]

    @property
    def memory(self):
        return self._memory if self._memory is not None else Vector(0)
    
    @memory.setter
    def memory(self, memory):
        self._memory = np.ndarray(memory)