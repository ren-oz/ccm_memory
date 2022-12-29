from abstract import AbstractMemory
from models import ChainedMatrixMemory, AttentionMemory
from hrrlib import HRR, rand

class SAMNetwork(AbstractMemory):
    def __init__(self, dimension: int, *args, **kwargs) -> None:
        self.state = HRR.identity(dimension)
        self.dimension = dimension
        self.memory = ChainedMatrixMemory(AttentionMemory, *args, **kwargs)
        self._associations = dict()
        self._symbols = {
            None: {
                'hrr': HRR.identity(dimension),
                'count': None,
            }
        }
        # self.add(None, None, None)  # Do nothing transition function

    def add(self, input:str, state:str, output:str) -> None:
        triple = (input, state, output) 
        if triple in self._associations.keys():
            return
        keys = self._symbols.keys()
        for s in triple:
            if s is not None:
                if s in keys:
                    self._symbols[s]['count'] += 1
                else:
                    self._symbols[s] = {
                        'hrr': rand.unitary(self.dimension), #, self._hkwargs), 
                        'count': 1,
                    }                
        symbol_in = self._symbols[input]['hrr'] * self._symbols[state]['hrr']
        symbol_out = self._symbols[output]['hrr'] 
        self.memory.add(symbol_in.as_real, symbol_out.as_real)
        self._associations[triple] = len(self.memory)-1
        
    def delete(self, input:str, state:str, output:str) -> None:
        triple = (input, state, output)
        if triple in self._associations.keys():
            index = self._associations[triple]
            self.memory.delete(index)
            self._associations.pop(triple)

            for s in triple:
                if s is not None:
                    self._symbols[s]['count'] -= 1
                    if not self._symbols[s]['count']:
                        self._symbols.pop(s)
    
    def retrieve(self, input: HRR) -> HRR:
        symbol_in = (input * self.state).as_real
        self.state = HRR(self.memory.retrieve(symbol_in))
        return self.state

    def symbol(self, key:str) -> HRR:
        s = self._symbols[key]
        return s['hrr']

    def probe_symbols(self, probe:HRR, top=3):
        return sorted([(key, HRR.similarity(probe, value['hrr'])) for key, value in self._symbols.items()], key=lambda x:x[1], reverse=True)[:top]

    def process_sequence(self, sequence:list) -> list:
        result = []
        for item in sequence:
            hrr = self.symbol(item)
            echo = self.retrieve(hrr)
            result.append(self.probe_symbols(echo, top=1)[0][0])
        return result
