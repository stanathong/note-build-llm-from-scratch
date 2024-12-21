# torch.nn.Parameters

* `torch.nn.Parameter` is a type of tensor that is intended to be trainable parameters in neural networks.
* When a tensor is defined as `torch.nn.Parameter`, it is automatically added to the list of the module's parameters.
* This is essential for making it accessible to optimizers during training.

## Key Properties

### Trainable

* `torch.nn.Parameter` is designed to be part of the model's learnable parameters, so optimizers like `torch.optim.SGD` or `torch.optim.Adam` will update its value during training.

### Automatic Registration

* When you define a torch.nn.Parameter inside a torch.nn.Module subclass, it is automatically registered as part of the module. This makes it easier to keep track of all trainable parameters in the model.

### Wrapper around Tensors

* A `Parameter` is essentially a `torch.Tensor` with an additional flag indicating that it is a learnable parameter. It behaves like a tensor but is treated specially by PyTorch's nn.Module.

## Illustration

```
import torch

class ExampleModel(torch.nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        # Define a trainable paramter
        self.weight1 = torch.nn.Parameter(torch.rand(4,2))
        self.bias1 = torch.nn.Parameter(torch.zeros(2))
        self.weight2 = torch.randn(3,2)
    
    def forward(self, x):
        return torch.matmul(x, self.weight2) + self.bias1
    
# Create a model instance
model = ExampleModel()

print('Trainable parameters:')
for name, param in model.named_parameters():
    print(f'Name: {name}, Size: {param.size()}, Requires Grad: {param.requires_grad}')
```

* As for output, there only weight1 and bias1 that are included in model.named_parameters() while weight2 is just a class member variable.
```
Name: weight1, Size: torch.Size([4, 2]), Requires Grad: True
Name: bias1, Size: torch.Size([2]), Requires Grad: True
```
