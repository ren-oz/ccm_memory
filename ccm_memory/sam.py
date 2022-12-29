from ccm_memory.abstract import AbstractMemory
from ccm_memory.models import ChainedMatrixMemory, ModernHopfield
from hrrlib import HRR, rand


class SAMNetwork(AbstractMemory):
    def __init__(self, dimension: int, *args, **kwargs) -> None:
        self.state = HRR.identity(dimension)
        self.dimension = dimension
        self.memory = ChainedMatrixMemory(ModernHopfield, *args, **kwargs)
        self._associations = dict()
        self._symbols = {
            None: {
                'hrr': HRR.identity(dimension),
                'count': None,
            }
        }

    def add(self, input:str, state:str, output:str) -> None:
        input, state, output = self._process_inputs(input, state, output)
        triple = (input, state, output)
        if triple in self._associations.keys():
            _f = True
        else:
            _f = False
            keys = self._symbols.keys()
            for s in triple:
                if s is not None:
                    if s in keys:
                        self._symbols[s]['count'] += 1
                    else:
                        self._symbols[s] = {
                            'hrr': rand.unitary(self.dimension), 
                            'count': 1,
                        }                
        symbol_in = self._symbols[input]['hrr'] * self._symbols[state]['hrr']
        symbol_out = self._symbols[output]['hrr'] 
        self.memory.add(symbol_in.as_real, symbol_out.as_real)
        if _f:
            self._associations[triple].append(len(self.memory)-1)
        else:
            self._associations[triple] = [len(self.memory)-1]
        
    def delete(self, input:str, state:str, output:str) -> None:
        input, state, output = self._process_inputs(input, state, output)
        triple = (input, state, output)
        if triple in self._associations.keys():
            indices = self._associations[triple].copy()
            for i in range(len(indices)):
                index = self._associations[triple][i]
                self.memory.delete(index-i)  # this should be fine since the list of indices is ordered
            self._associations.pop(triple)
            # Adjust all other indices
            for a in self._associations:
                for j in range(len(self._associations[a])):
                    num = self._associations[a][j]
                    for index in indices:
                        if self._associations[a][j] > index:
                            num -= 1
                    self._associations[a][j] = num

            for s in triple:
                if s is not None:
                    self._symbols[s]['count'] -= 1
                    if not self._symbols[s]['count']:
                        self._symbols.pop(s)
    
    def retrieve(self, input: HRR) -> HRR:
        symbol_in = (input * self.state).as_real
        self.state = HRR(self.memory.retrieve(symbol_in))
        return self.state

    def symbol_hrr(self, key:str) -> HRR:
        s = self._symbols[key]
        return s['hrr']

    def probe_symbols(self, probe:HRR, top=3) -> list:
        return sorted([(key, HRR.similarity(probe, value['hrr'])) for key, value in self._symbols.items()], key=lambda x:x[1], reverse=True)[:top]

    def process_sequence(self, sequence:list) -> list:
        result = []
        sequence = self._process_inputs(*sequence)
        for item in sequence:
            hrr = self.symbol(item)
            echo = self.retrieve(hrr)
            result.append(self.probe_symbols(echo, top=1)[0][0])
        return result
    
    def _process_inputs(self, *args) -> list:
        result = []
        for arg in args:
            if arg is None:
                result.append(arg)
            else:
                result.append(str(arg))
        return result
    
    @property
    def associations(self) -> list:
        return list(self._associations.keys())
    
    @property
    def symbols(self) -> list:
        return list(self._symbols.keys())