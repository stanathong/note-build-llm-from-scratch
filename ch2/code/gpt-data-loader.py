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

data_loader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(data_loader) # Convert dataloader into a Python iterator
                              # to fetch the next entry via 
                              # Python's built-in next() function
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

'''
first_batch 
[tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
second_batch
[tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
'''

# Setting different batch size and strides
data_loader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print('Inputs:\n', inputs)
print('\nTargets:\n', targets)

'''
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
'''
