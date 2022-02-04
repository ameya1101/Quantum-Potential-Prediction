from torch.utils.data import Dataset

class CoordinateData(Dataset):
    '''
    Class to generate a PyTorch Dataset for the training coordinates.
    '''
    def __init__(self, data, N) -> None:
        super(CoordinateData, self).__init__()
        self.data = data
        self.len = N
    
    def __getitem__(self, index):
        return self.data[index]

    
    def __len__(self):
        return self.len
        