import torch

class KernelLoader:
    def __init__(self, directory):
        self.directory = directory

    def __getitem__(self, index):
        filename = f"{self.directory}/{index}.pth"
        tensor = torch.load(filename)
        return tensor