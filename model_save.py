import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.liner1 = nn.Linear(1,1)
    
    def forward(self,input):
        return self.liner1(input)
        
net = Model()
x = torch.tensor([1.0])
print(net(x))
# method 1
#保存结构和参数
torch.save(net, "net.pth")

#读取模型
net_load = torch.load("net.pth")
print(net_load)

# method 2 
#保存参数
torch.save(net.state_dict() , "net_state.pth")

#读取模型
model = Model()
model.load_state_dict(torch.load("net_state.pth"))
print(model)
