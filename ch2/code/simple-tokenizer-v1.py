import re
import urllib.request
import textwrap

class SimpleTokenizerV1:
    '''
    A simple tokenizer that encodes and decodes text based on a vocabulary.
    Attributes:
        str_to_int (dict): Mapping from token string to token id
        int_to_str (dict): Mapping from token id to token string
    '''
    def __init__(self, vocab):
        '''
        Initalises the tokeniser with a given vocabulary
        
        Args:
            vocab (dict): A dictionary mapping token string:id
        '''
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        '''
        Process input text into token id.
        
        Args:
            text (str): The text to encode
        ''' 
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        try:
            ids = [self.str_to_int[token] for token in preprocessed] 
        except KeyError as e:
            raise ValueError(f'Token \'{e.args[0]}\' not found in vocabulary.') from e
        return ids
    
    def decode(self, ids):
        '''
        Converts token ids back into text

        Args:
            ids (list): A list of integer ids to decode
        '''
        try:
            text = ' '.join([self.int_to_str[i] for i in ids])
        except KeyError as e:
            raise ValueError(f'ID \'{e.args[0]}\' not found in vocabulary.') from e

        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
def main():
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
           "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words) # 1130
    print(f'The total number of words in the vocabulary is {vocab_size}')

    vocab = {token:integer for integer, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab)

    # valid text
    try:
        #text =  '''
        #        I found the couple at tea beneath their palm-trees; 
        #        and Mrs. Gisburn's welcome was so genial that, in the 
        #        ensuing weeks, I claimed it frequently. It was not that 
        #        my hostess was "interesting": 
        #        '''
                
        text =  '''
                "My dear, since I've chucked painting people don't say 
                that stuff about me--they say it about Victor Grindle,"
                '''
        ids = tokenizer.encode(textwrap.dedent(text))
        decoded_text = tokenizer.decode(ids)
        print(decoded_text)
    except ValueError as e:
        print('Error: ', e)

    # invalid text
    try:
        text =  '''
                There is a celebrity hippo family in Thailand. 
                My favourite hippo is Moo Deng.
                '''
        ids = tokenizer.encode(textwrap.dedent(text))
        decoded_text = tokenizer.decode(ids)
        print(decoded_text)
    except ValueError as e:
        print('Error: ', e)
        # Error:  Token 'celebrity' not found in vocabulary.

if __name__ == '__main__':
    main()

