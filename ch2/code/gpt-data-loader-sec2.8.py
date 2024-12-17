import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import urllib.request

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # Tokenize the entire text
        
        # sliding window of max_lenght size with stride
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] # shift by 1
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
# Load the inputs in batches via Pytorch DataLoader
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Instantiate the BPE tokenizer from tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, # if True, drops the last batch if it is shorter
                             # than the specified batch_size to prevent
                             # loss spikes during training
        num_workers=num_workers # The number of CPU processes for preprocessing
    )
    return dataloader

# Test
# Download the text file to be used for training
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/"
       "heads/main/ch02/01_main-chapter-code/the-verdict.txt")
file_name = "the-verdict.txt"
urllib.request.urlretrieve(url, file_name)

with open(file_name, 'r', encoding='utf-8') as f:
    raw_text = f.read()

### Using max_length = 4, batch_size=8, stride=4

max_length = 4
data_loader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(data_loader) # Convert dataloader into a Python iterator
                              # to fetch the next entry via 
                              # Python's built-in next() function
inputs, targets = next(data_iter)
print('Token IDs:\n', inputs)
print('\nInputs shape:\n', inputs.shape)

# The token ID tensor is 8x4 dimensional meaning that 
# the data batch consists of 8 text samples with 4 tokens each.
'''
Token IDs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Inputs shape:
 torch.Size([8, 4])
'''

### Use the embedding layer to embedded these token IDs into 256 dimensional vectors

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) # torch.Size([8, 4, 256])

# The 8 x 4 x 256-dimensional tensor output shows that each token ID is now embedded as a 256-dimensional vector.

# Absolute embedding approach
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) 
                                    # place holder
                                    # a sequence of 0..contex_length-1
print(pos_embeddings.shape) # torch.Size([4, 256])

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape) # torch.Size([8, 4, 256])

