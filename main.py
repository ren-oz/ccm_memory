from hrrlib import HRR, rand
from abstract import AbstractMemoryItem, AbstractMemory, ABC, abstractmethod
import numpy as np
from utils import softmax
from typing import Union


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


class AttentionMemory(AbstractMatrixMemory):

    def __init__(self, matrix:np.ndarray=None, beta:float=1):
        super().__init__(matrix=matrix)
        self._beta = beta

    def activation(self, probe: Vector) -> Activation:
        dot = self._memory@probe
        return softmax(self._beta*dot).view(Activation)

    def apply_activation(self, activation: Activation) -> Vector:
        return (self._memory.T@activation).view(Vector)


class MINERVA2(AbstractMatrixMemory):

    def __init__(self, matrix:np.ndarray=None, exponent=3):
        super().__init__(matrix=matrix)
        self._exponent = exponent

    def activation(self, probe: Vector) -> Activation:
        probe = np.array(probe)
        # find the max number of non-zero entries between probe and memory item
        n = [
            max(np.where(self._memory[i,:] == 0)[0].size, np.where(probe == 0)[0].size) 
            for i in range(self._memory.shape[0])
        ]
        scale = (np.ones(len(self._memory))*self._memory.shape[1])-n
        dot_scaled = (self._memory@probe)/scale
        return (dot_scaled**self._exponent).view(Activation)

    def apply_activation(self, activation: Activation) -> Vector:
        return (self._memory.T@activation).view(Vector)