from importlib.metadata import version
import tiktoken

# Check version of tiktoken
print('tiktoken version: ', version('tiktoken')) # 0.8.0

# Once installed, we can instantiate the BPE tokenizer from tiktoken as follows:
tokenizer = tiktoken.get_encoding('gpt2') 

text = (
    'My favourite hippo is Moo Deng or MooDeng. <|endoftext|> Structure from Motion '
    'and SLAM are both used to map the scene.'
)

integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print('Type of integers: ', type(integers)) # <class 'list'>
print(integers)

# Convert the token id back into text using decode method
strings = tokenizer.decode(integers)
print(strings)

'''
[3666, 12507, 18568, 78, 318, 337, 2238, 41985, 393, 337, 2238, 35, 1516, 13, 220, 50256, 32522, 422, 20843, 290, 12419, 2390, 389, 1111, 973, 284, 3975, 262, 3715, 13]
My favourite hippo is Moo Deng or MooDeng. <|endoftext|> Structure from Motion and SLAM are both used to map the scene.
'''

# Test BPE tokenizer on the unknown words: Akwirw ier

text = 'Akwirw ier'
integers = tokenizer.encode(text)
strings = tokenizer.decode(integers)
print(text)
print(integers)
print(strings) 

