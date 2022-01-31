import torch.nn as nn

class MPNN(nn.Module):
    def __init__(self, in_size: int, out_scaling: float, delta: float = 0.01):
        super(MPNN, self).__init__()
        self.layer1 = nn.Linear(in_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 1)
        self.out_scaling = out_scaling

        # Scale the model weights by delta
        for param in self.parameters():
            param = param * delta
    
    def forward(self, x):
        h = self.layer1(x)
        h = nn.ReLU(h)
        h = self.layer2(h)
        h = nn.ReLU(h)
        h = self.layer3(h)
        h = nn.ReLU(h)
        h = nn.ReLU(self.layer4(h)) + h
        out = self.layer5(h)
        return self.out_scaling * out