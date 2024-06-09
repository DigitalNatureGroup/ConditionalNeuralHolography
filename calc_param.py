import torch
import torch.nn as nn
from holonet import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# パラメータ数の計算
complex_params = count_parameters(InitialDoubleUnet(6, 16))
encoder_params=count_parameters(FinalPhaseOnlyUnet(8, 32, num_in=2))

print(complex_params)
print(encoder_params)
