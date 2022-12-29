from ccm_memory.sam import SAMNetwork
import random

sam = SAMNetwork(200, beta=5)

# Add associations to memory
sam.add('espresso', 'summer', 'vacation')       # espresso + summer = vacation
sam.add('umbrella', 'vacation', 'beach')        # umbrella + vacation = beach

sam.add('espresso', 'winter', 'family')         # espresso + winter = family
sam.add('umbrella', 'family', 'Mary Poppins')   # umbrella + family = Mary Poppins

# Set initial state
INITIAL_STATE = random.choice(['summer', 'winter'])
print("Initial state:", INITIAL_STATE)
sam.state = sam.symbol_hrr(INITIAL_STATE)

# Define input sequence
input_seq = ['espresso', 'umbrella']

# Process sequence with model
result = sam.process_sequence(input_seq)

# Show evolution of model state
print(result)