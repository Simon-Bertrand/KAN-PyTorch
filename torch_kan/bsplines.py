from torch import nn
import torch
class BatchedBSplines(nn.Module):
    """
    Batched B-Splines module for evaluating B-spline interpolation on batches of data.

    Args:
      inDim (int): Number of control points
      k (int): Degree of B-spline.

    Attributes:
      t (torch.Tensor): Padded x-axis knots (inDim+2*k, ).
    """

    def __init__(self, inDim: int, k: int):
        super().__init__()
        if not isinstance(k, int):
            raise ValueError(
                f"The degree k of B-spline must be an integer. Current : {k}"
            )
        self.k = k
        if not isinstance(inDim, int):
            raise ValueError(
                f"The number of control points (inDim) must be an integer. Current : {inDim}"
            )
        self.m = inDim - 1
        self.t = torch.arange(-self.k, self.m + self.k) / (self.m - 1)

    def forward(self, x: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method to compute batched B-spline interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, nEval).
            cp (torch.Tensor): Control points tensor of shape (B, inDim).

        Returns:
            torch.Tensor: Evaluated B-spline interpolation of shape (B, nEval).
        """
        assert (
            cp.size(-1) == self.m + 1
        ), "Control points must have m+1 element in the last dimension"
        assert cp.size(-2) == x.size(
            -2
        ), "Both control points and x batch sizes must equals"
        # Pads the control points with k-1 last elements
        paddedCp = torch.hstack([cp, cp[..., -1:].expand(-1, self.k - 1)])  # (B, m+k)
        # Gets the bin indices that contains x
        leftRange = (self.t.unsqueeze(-2) > x.clamp(0, 1).unsqueeze(-1)).float().argmax(
            -1
        ) - 1
        # Create batched indices slices : B times a corresponding slice(m-k, m+1)
        slicesIndices = leftRange.unsqueeze(-1) + torch.arange(-self.k, 1).unsqueeze(-2)
        # Retrieve from control points all the batched slices
        d = paddedCp.gather(-1, slicesIndices.flatten(-2)).unflatten(
            -1, (slicesIndices.shape[-2:])
        )
        # Proceed to optimized batched de Boor algorithm
        for r in range(1, self.k + 1):
            for j in range(self.k, r - 1, -1):
                alphas = (x - self.t[j + leftRange - self.k]) / (
                    self.t[j + 1 + leftRange - r] - self.t[j + leftRange - self.k]
                )
                d[..., j] = (1.0 - alphas) * d[..., j - 1] + alphas * d[..., j]
        return d[..., self.k]