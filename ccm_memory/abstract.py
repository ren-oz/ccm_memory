from abc import ABC, abstractmethod

class AbstractMemoryItem(ABC): ...

class AbstractMemory(ABC):

    @abstractmethod
    def add(self, item: AbstractMemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, item: AbstractMemoryItem) -> AbstractMemoryItem: ...

    @abstractmethod
    def delete(self, x: object) -> None: ...