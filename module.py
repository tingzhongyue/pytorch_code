import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,input):
        output = input + 1 
        return output 
        
net = Model()
x = torch.tensor(1.0)
print(net(x))