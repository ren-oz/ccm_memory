from ccm_memory.sam import SAMNetwork

sam = SAMNetwork(200, beta=1)

# Add conflicting associations to memory
sam.add('a', None, 'y')
sam.add('a', None, 'n')

# Add branches
sam.add('b', 'y', 'yes')
sam.add('b', 'n', 'no')

input_seq = ['a', 'b']
results = sam.process_sequence(input_seq)

# Show evolution of model state
print(results)

# Add conflicting associations to memory
sam.add('a', None, 'y')

# Reset state
sam.state = sam.symbol_hrr(None)

input_seq = ['a', 'b']
results = sam.process_sequence(input_seq)

# Show evolution of model state
print(results)