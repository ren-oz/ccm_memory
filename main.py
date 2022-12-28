from hrrlib import HRR, rand
from abstract import AbstractMemoryItem, AbstractMemory
import numpy as np
from utils import softmax
from typing import Union


class Vector(np.ndarray, AbstractMemoryItem): ...


class Activation(Vector): ...


class MatrixMemory(AbstractMemory):
    def __init__(self, matrix:np.ndarray=None):
        self._memory = matrix
    
    def add(self, item: Vector) -> None:
        v = np.array(item)
        if self._memory is None:
            self._memory = v.reshape((1, v.size))
        else:
            self._memory = np.concatenate((self._memory, v.reshape((1, v.size))))
    
    def retrieve(self, item: AbstractMemoryItem) -> AbstractMemoryItem:
        # some kind of check up the inheritance hierarchy to determine which error msg to display
        err = f'Must implement this function in class "{self.__class__.__name__}"'
        raise NotImplementedError(err)
        
    def delete(self, index:int) -> None:
        self._memory = self._memory[[i for i in range(self._memory.shape[0]) if i != index]]

    @property
    def memory(self):
        return self._memory if self._memory is not None else Vector(0)
    
    @memory.setter
    def memory(self, memory):
        self._memory = np.ndarray(memory)


class AttentionMemory(MatrixMemory):

    def __init__(self, matrix:np.ndarray=None, beta:float=1):
        super().__init__(matrix=matrix)
        self._beta = beta

    def activation(self, probe: Vector) -> Activation:
        dot = self._memory@probe
        return softmax(self._beta*dot)

    def retrieve(self, activation: Activation) -> Vector:
        return self._memory.T@activation