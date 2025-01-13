import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # Implement 5 layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
                          GELU()),                                                    
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Compute output of the current layer
            layer_output = layer(x)
            print(f'layer#{i} : x = {x.shape}, output = {layer_output.shape}')
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    '''
    layer#0 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#1 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#2 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#3 : x = torch.Size([1, 3]), output = torch.Size([1, 3])
    layer#4 : x = torch.Size([1, 3]), output = torch.Size([1, 1])
    '''
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
            (x + 0.044715 * torch.pow(x,3))) 
        )

# Utility function to compute gradients in the model's backward pass
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    # calculate loss based on how close the target and the output
    loss = nn.MSELoss()
    loss = loss(output, target)
    # calculate gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0, -1.]])

# Without shortcut connections
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)

'''
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.00012011159560643137
layers.2.0.weight has gradient mean of 0.0007152039906941354
layers.3.0.weight has gradient mean of 0.0013988736318424344
layers.4.0.weight has gradient mean of 0.005049645435065031
'''

# With shortcut connections
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_without_shortcut, sample_input)

'''
layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694106817245483
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732204914093
layers.4.0.weight has gradient mean of 1.3258540630340576
'''