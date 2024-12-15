from importlib.metadata import version
import tiktoken
import re
import urllib.request

# Check version of tiktoken
print('tiktoken version: ', version('tiktoken')) # 0.8.0

# Instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding('gpt2')

# Get the URL to the 'the-verdict'
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
        "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_name = 'the-verdict.txt'
urllib.request.urlretrieve(url, file_name)

with open(file_name, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Tokenize the whole document
enc_text = tokenizer.encode(raw_text)
print('Len of enc_text = ', len(enc_text)) # 5145

# Remove the first 50 tokens as its result is more *interesting*
enc_sample = enc_text[50:]

# Create the input-target pairs: x, y
# x is the input, y is the target tokens -- the input shifted by 1
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f'x: {x}')
print(f'y:      {y}')
'''
x: [290, 4920, 2241, 287]
y:      [4920, 2241, 287, 257]
'''

# Create the next-word prediction tasks
for i in range (1, context_size+1):
    context = enc_sample[:i] # [0,i)
    desired = enc_sample[i]
    print(context, '---->', desired)

'''
[290] ----> 4920
[290, 4920] ----> 2241
[290, 4920, 2241] ----> 287
[290, 4920, 2241, 287] ----> 257
'''

# Create the next-word prediction tasks
# This time, convert from token ID into text
for i in range (1, context_size+1):
    context = enc_sample[:i] # [0,i)
    desired = enc_sample[i]
    print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))
'''
 and ---->  established
 and established ---->  himself
 and established himself ---->  in
 and established himself in ---->  a
'''

'''
Note that, without using [] resulting in compiler error
print(tokenizer.decode(context), '---->', tokenizer.decode(desired))
Traceback (most recent call last):
    print(tokenizer.decode(context), '---->', tokenizer.decode(desired))
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: argument 'tokens': 'int' object cannot be converted to 'Sequence'
'''



