from abc import ABC, abstractmethod

class AbstractMemoryItem(ABC): ...

class AbstractMemory(ABC):

    @abstractmethod
    def add(self, item: AbstractMemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, cue: AbstractMemoryItem) -> AbstractMemoryItem: ...

    @abstractmethod
    def delete(self, item: AbstractMemoryItem) -> None: ...