from ccm_memory.sam import SAMNetwork
import random

sam = SAMNetwork(200, beta=50)

n_keys = 10
valid_key = random.choice(range(n_keys))

# Add associations to memory
for key in range(n_keys):
    if key == valid_key:
        sam.add(key, None, 'valid')
    else:
        sam.add(key, None, 'invalid')
    sam.add(key, 'valid', 'valid')
    sam.add(key, 'invalid', 'invalid')

######### Trial 1

# Try random keys (should be locked out indefinitely unless initial guess is correct)
input_len = 5
input_seq = [random.choice(range(n_keys)) for _ in range(input_len)]

print("Valid Key:", valid_key)
print("Input (random):", input_seq)

# Process sequence with model
result = sam.process_sequence(input_seq)

# Show evolution of model state
print("Output (random):", result)

######### Trial 2

# Reset SAM state
sam.state = sam.symbol_hrr(None)

# Start with valid key (should then accept for all inputs)
input_seq = [valid_key] + [random.choice(range(n_keys)) for _ in range(input_len-1)]

print("Input (valid):", input_seq)

# Process sequence with model
result = sam.process_sequence(input_seq)

# Show evolution of model state
print("Output (valid):", result)