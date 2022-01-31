from mimetypes import init
from torch.utils.data import Dataset

class CoordinateData(Dataset):
    def __init__(self, data) -> None:
        super(CoordinateData, self).__init__()
        self.data = data
    
    def __getitem__(self, index):
        return self.data[:, index]
    
    def __len__(self):
        return len(self.data)