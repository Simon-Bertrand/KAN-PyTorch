
from torch import nn
import torch

from .bsplines import BatchedBSplines
class KANLayer(nn.Module):
    CPS_INIT_STD = 0.1
    UPDATE_CPS_N_EVAL = 32

    def __init__(self, inDim, outDim, k, nCps):
        super(KANLayer, self).__init__()
        self.k = k
        self.inDim = inDim
        self.outDim = outDim
        self.silu = nn.SiLU()
        self.nCps = nCps
        self.splines = BatchedBSplines(self.nCps, self.k)
        self.cps = nn.Parameter(
            (torch.randn(self.inDim, self.outDim, self.nCps) * self.CPS_INIT_STD)
        )
        self.w = nn.Parameter(
            torch.randn(2, inDim, outDim, 1) * (2 / (inDim * outDim)) ** 0.5
        )

    def updateCps(self, newNCps):
        newNCps = max(newNCps,self.k+1) #Avoid having less control points than the degree of the B-splines.
        #Generate a linear grid of points to evaluate the splines.
        x = torch.linspace(*self.splines.X_RANGE, self.UPDATE_CPS_N_EVAL).unsqueeze(0).unsqueeze(0).expand(-1, self.inDim, -1)
        # Get the current splines curves evaluated for x
        B = self.splines(x, self.cps)  # (1, inDim, outDim, nEval).
        #Create new BatchedBSplines object with the new number of control points.
        newSplines = BatchedBSplines(newNCps, self.k)
        #Retrieve Bi,p(x) values.
        A = newSplines._bSplines(x)  # (1, inDim, nEval, nCps)
        #Solve the least square problem to find the new control points values.
        # min ||A * X - B||_{fro}
        newCps = torch.linalg.lstsq(A, B.moveaxis(-2, -1)).solution.moveaxis(-2, -1).squeeze(0)

        # Set the new values concerned by the control points.
        self.nCps = newNCps
        self.splines = newSplines
        self.cps = nn.Parameter(newCps)
        #Now KANLayer is updated with the new control points that were initialized with the least square solution of the previous control points.

    def forward(self, x):
        # x : (B, inDim)
        return (
            (
                self.w[0] * self.splines(x.unsqueeze(-1), self.cps)
                + self.w[1] * self.silu(x).unsqueeze(-1).unsqueeze(-1)
            )
            .sum(-3)
            .squeeze(-1)
        )