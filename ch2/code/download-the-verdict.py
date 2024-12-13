import urllib.request
import re # regular expression

# 1. Download the text file that will be used to train the LLM models
# url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
       "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 2. Load the 'the-verdict.txt' file using Python's standard file reading utilities
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(f'Total number of characters: {len(raw_text)}')
print(raw_text[:5]) # I HAD
print(raw_text[:50]) # I HAD always thought Jack Gisburn rather a cheap g
'''
012345678901
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow 
'''

# 3. Use python's regular expression library to split text on white space char
# import re
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'(\s)', text)
print(splitted)
# The result is a list of individual words, whitespaces and punctuation characters
'''
['I', ' ', 'HAD', ' ', 'always', ' ', 'thought', ' ', 'Jack', ' ', 'Gisburn', 
' ', 'rather', ' ', 'a', ' ', 'cheap', ' ', 'genius--though', ' ', 'a', ' ', 
'good', ' ', 'fellow', ' ', 'enough--so', ' ', 'it', ' ', 'was', ' ', 'no', 
' ', '']
'''

# 4. To avoid the case where some words are still conntected to punctuation characters.
# Modify the regular expression splits on white space \s, comman and period
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'([,.-]|\s)', text)
print(splitted)
'''
['I', ' ', 'HAD', ' ', 'always', ' ', 'thought', ' ', 'Jack', ' ', 'Gisburn', 
' ', 'rather', ' ', 'a', ' ', 'cheap', ' ', 'genius', '-', '', '-', 'though', 
' ', 'a', ' ', 'good', ' ', 'fellow', ' ', 'enough', '-', '', '-', 'so', ' ', 
'it', ' ', 'was', ' ', 'no', ' ', '']
'''

# 5. Remove white space characters
splitted = [item for item in splitted if item.strip()]
print(splitted)
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
'genius', '-', '-', 'though', 'a', 'good', 'fellow', 'enough', '-', '-', 'so', 
'it', 'was', 'no']
'''

# 6. Remove extra special chars
text = raw_text[:99]
print(text)
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
splitted = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(splitted)
splitted = [item.strip() for item in splitted if item.strip()]
print(splitted)
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 
'was', 'no']
'''

# 7. Process the whole file
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # 4096
print(preprocessed[:30]) # the first 30 tokens
'''
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 
 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 
 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
'''

# 8. Create a list of all unique tokens and sort them alphabetically to determine the vocab size
all_words = sorted(set(preprocessed))
vocab_size = len(all_words) # 1130
print(f'The total number of words in the vocabulary is {vocab_size}')

# 9. Create vocabulary by mapping id to token
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(f'{i} : {item}')
    if i >= 50:
        break

'''
0 : ('!', 0)
1 : ('"', 1)
2 : ("'", 2)
3 : ('(', 3)
4 : (')', 4)
5 : (',', 5)
6 : ('--', 6)
7 : ('.', 7)
8 : (':', 8)
9 : (';', 9)
10 : ('?', 10)
11 : ('A', 11)
12 : ('Ah', 12)
13 : ('Among', 13)
...
'''
