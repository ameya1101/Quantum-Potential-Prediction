import torch

class MPNN(torch.nn.Module):
    '''
    Implements the network architecture for the Metropolis Neural Network. 

    Inputs
    ------
        in_size: int
            Input dimension
        out_scaling: float
            Factor to scale the network output
        delta: float
            Factor to scale the network weights at initialization
    '''
    def __init__(
        self, in_size: int, 
        out_scaling: float, final_act=None,
        delta: float = 0.01
    ) -> None:
        super(MPNN, self).__init__()
        self.layer1 = torch.nn.Linear(in_size, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 128)
        self.layer4 = torch.nn.Linear(128, 128)
        self.layer5 = torch.nn.Linear(128, 1)
        self.out_scaling = out_scaling
        self.final_act = final_act

        # Scale the model weights by delta
        for param in self.parameters():
            param = param * delta
    
    def forward(self, x) -> torch.Tensor:
        h = self.layer1(x)
        h = torch.nn.functional.relu(h)
        h = self.layer2(h)
        h = torch.nn.functional.relu(h)
        h = self.layer3(h)
        h = torch.nn.functional.relu(h)
        h = torch.nn.functional.relu(self.layer4(h)) + h
        if self.final_act == 'sigmoid':
            out = torch.sigmoid(self.layer5(h))
        else:
            out = self.layer5(h)
        return self.out_scaling * out
        