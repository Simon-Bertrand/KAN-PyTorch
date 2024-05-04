from typing import List
from torch import nn

from .kan_layer import KANLayer
class KAN(nn.Module):
    def __init__(self, dims : List[int], k : int, nCps : int):
        super(KAN, self).__init__()
        self.dims = dims
        self.layers = nn.ModuleList(
            [
                KANLayer(inDim,outDim,k,nCps) for inDim,outDim in zip(self.dims[:-1],self.dims[1:])
            ]
        )
        self.kan = nn.Sequential(*self.layers)
    def forward(self, x):
        # x : (B, inDim)
        return self.kan(x)