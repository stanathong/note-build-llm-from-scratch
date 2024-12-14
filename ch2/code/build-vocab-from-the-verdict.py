import urllib.request
import re # regular expression

# 1. Download the text file that will be used to train the LLM models
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
       "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_name = "the-verdict.txt"
urllib.request.urlretrieve(url, file_name)

# 2. Load the 'the-verdict.txt' file using Python's standard file reading utilities
with open(file_name, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f'Total number of characters: {len(raw_text)}')
# Total number of characters: 20479
print(raw_text[:10]) # I HAD alwa
print(raw_text[-10:]) # d of art."

# 3. Preprocess the file by using python's regular expression lib 
#    - to split text on while space char \s
#    - to split words connected to special chars e.g. punctuation, comma, etc
#    - to split text using ,.:;?_!"()\', --, space (\s)
preprocessed = re.split(r'[,.:;?_!"()\']|--|\s', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f'Total number of all tokens: {len(preprocessed)}') #  3788
print(preprocessed[:10]) # The first 10 tokens

# 4. Create a list of all unique tokens and sort them alphabetically
all_tokens = sorted(set(preprocessed))
vocab_size = len(all_tokens)
print(f'Total number of unique tokens: {vocab_size}') # 1118

# 5. Extend the set of tokens to include <|unk|> and <|endoftext|>
all_tokens.extend(['<|unk|>', '<|endoftext|>'])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(f'Total number of vocab tokens: {len(vocab)}') # 1120
# Print the last 5 items
for id, item in list(vocab.items())[-5:]:
    print(f'{id} : {item}')


'''
Output:
-------
Total number of characters: 20479
I HAD alwa
d of art."
Total number of all tokens: 3788
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius']
Total number of unique tokens: 1118
Total number of vocab tokens: 1120
younger : 1115
your : 1116
yourself : 1117
<|unk|> : 1118
<|endoftext|> : 1119
'''
