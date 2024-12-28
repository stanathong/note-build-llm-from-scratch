import torch

# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print('a:\n', a)
print('a.transpose(2, 3):\n', a.transpose(2, 3))

# a:        (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4) @
# a.t(2,3): (b, num_heads, head_dim, num_tokens) = (1, 2, 4, 3)
#      = (b, num_heads, num_tokens, num_tokens)
print(a @ a.transpose(2, 3))

'''
a.transpose(2, 3):
 tensor([[[[0.2745, 0.8993, 0.7179],
          [0.6584, 0.0390, 0.7058],
          [0.2775, 0.9268, 0.9156],
          [0.8573, 0.7388, 0.4340]],

         [[0.0772, 0.4066, 0.4606],
          [0.3565, 0.2318, 0.5159],
          [0.1479, 0.4545, 0.4220],
          [0.5331, 0.9737, 0.5786]]]])
'''

'''
(1, 2, 3, 4) x (1, 2, 4, 3) = (1, 2, 3, 3)
a @ a.transpose(2, 3)
tensor([[[[1.3208, 1.1631, 1.2879],
          [1.1631, 2.2150, 1.8424],
          [1.2879, 1.8424, 2.0402]],

         [[0.4391, 0.7003, 0.5903],
          [0.7003, 1.3737, 1.0620],
          [0.5903, 1.0620, 0.9912]]]])
'''

# If we were to compute the matrix multiplication for each head separately
first_head = a[0,0,:,:]
first_res = first_head @ first_head.T
print('First head:\n', first_res)

second_head = a[0,1,:,:]
second_res = second_head @ second_head.T
print('Second head:\n', second_res)

'''
First head:
 tensor([[1.3208, 1.1631, 1.2879],
        [1.1631, 2.2150, 1.8424],
        [1.2879, 1.8424, 2.0402]])

Second head:
 tensor([[0.4391, 0.7003, 0.5903],
        [0.7003, 1.3737, 1.0620],
        [0.5903, 1.0620, 0.9912]])        
'''

