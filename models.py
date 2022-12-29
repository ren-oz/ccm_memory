from base import AbstractMatrixMemory, Activation, Vector, MemoryPair
from abstract import AbstractMemory
import numpy as np
from utils import softmax

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


class ChainedMatrixMemory(AbstractMatrixMemory):
    # Required structure for Stateful attention memory implementations
    def __init__(self, memory_model: type, *args, **kwargs):
        super().__init__()
        self._memory = MemoryPair(memory_model, *args, **kwargs)

    def activation(self, probe: Vector) -> Activation:
        return self._memory[0].activation(probe)

    def apply_activation(self, activation: Activation) -> Vector:
        return self._memory[1].apply_activation(activation)

    def add(self, a:Vector, b:Vector) -> None:
        self._memory[0].add(a)
        self._memory[1].add(b)

    def delete(self, index:int) -> None:
        self._memory[0].delete(index)
        self._memory[1].delete(index)

    def __len__(self):
        return self._memory[0].__len__()
