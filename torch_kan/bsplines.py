from torch import nn
import torch

#VERSION 2 - Version operating on the two last dimensions : We define a matrix of function
class BatchedBSplines(nn.Module):
    """
    Batched B-Splines module for evaluating B-spline interpolation on batches of data.

    Args:
      nCps (int): Number of control points
      k (int): Degree of B-spline.

    Attributes:
      t (torch.Tensor): Padded x-axis knots (inDim+2*k, ).
    """

    def __init__(self, nCps: int, k: int):
        super().__init__()
        if not isinstance(k, int):
            raise ValueError(
                f"The degree k of B-spline must be an integer. Current : {k}"
            )
        self.k = k
        if not isinstance(nCps, int):
            raise ValueError(
                f"The number of control points (inDim) must be an integer. Current : {nCps}"
            )
        self.m = nCps - 1
        self.t = torch.arange(-self.k, self.m + self.k) / (self.m - 1)


    def forward(self, x: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method to compute batched B-spline interpolation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, inDim, nEval).
            cp (torch.Tensor): Control points tensor of shape (inDim, outDim, nCpts).

        Returns:
            torch.Tensor: Evaluated B-spline interpolation of shape (B, inDim, outDim, nEval).
        """
        assert x.ndim ==3 and cp.ndim==3, f"Input x and control points cp must have 3 dimensions : Current {x.ndim=} and {cp.ndim=}"
        assert (
            cp.size(-1) == self.m + 1
        ), f"Control points must have nCps element in the last dimension. Current {cp.size(-1)=} and {self.m+1=}"
        assert (
            x.size(1) == cp.size(0)
        ), f"inDim must equal for cp and input x. Current {x.size(1)=} and {cp.size(0)=}"
        B = x.size(0)
        outDim = cp.size(1)

        # Pads the control points with k-1 times the last element
        paddedCp = (
            torch.cat(
             [cp]+(self.k - 1)*[cp[..., -1:]], dim=-1
            )  # (inDim, outDim, nCpts + k-1)
            .unsqueeze(0) # (1, inDim, outDim, nCpts + k-1)
            .expand(B,-1,-1,-1) # (B, inDim, outDim, nCpts + k-1)
        )

        # Gets the bin indices that contains x
        leftRange = (
            self.t > x.clamp(0, 1).unsqueeze(-1) # (B, inDim, nEval, inDim+2*k)
        ).float().argmax(-1) - 1 # (B, inDim, nEval)

        # Create batched indices slices : B times a corresponding slice(m-k, m+1)
        slicesIndices = (
            (
                leftRange.unsqueeze(-1) + torch.arange(-self.k, 1) #  (B, inDim, nEval, k)
            )
            .unsqueeze(-3) #  (B, inDim, 1, nEval, k)
            .expand(-1, -1, outDim, -1, -1)  # (B, inDim, outDim, nEval, k)
        )

        # Retrieve from control points all the batched slices
        d = (
            paddedCp # (B, inDim, outDim, nCpts + k-1)
            .gather(-1, slicesIndices.flatten(-2)) # (B, inDim, outDim, nEval * k)
            .unflatten(-1,(slicesIndices.shape[-2:])) # (B, inDim, outDim, nEval,  k)
        )

        # Proceed to optimized batched de Boor's algorithm:
        # Ref : https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
        for r in range(1, self.k + 1):
            for j in range(self.k, r - 1, -1):
                alphas = (
                    (x - self.t[j + leftRange - self.k])
                    / (self.t[j + 1 + leftRange - r] - self.t[j + leftRange - self.k])
                ).unsqueeze(-2) # (B, inDim, 1, nEval)
                # d : (B, inDim, outDim, nEval,  k)
                d[..., j] = (1.0 - alphas) * d[..., j - 1] + alphas * d[..., j] #  (B, inDim, outDim, nEval)
        return d[..., self.k] #  (B, inDim, outDim, nEval)


#VERSION 1 - Version operating only on the last dimension
# class BatchedBSplines(nn.Module):
#     """
#     Batched B-Splines module for evaluating B-spline interpolation on batches of data.

#     Args:
#       inDim (int): Number of control points
#       k (int): Degree of B-spline.

#     Attributes:
#       t (torch.Tensor): Padded x-axis knots (inDim+2*k, ).
#     """

#     def __init__(self, inDim: int, k: int):
#         super().__init__()
#         if not isinstance(k, int):
#             raise ValueError(
#                 f"The degree k of B-spline must be an integer. Current : {k}"
#             )
#         self.k = k
#         if not isinstance(inDim, int):
#             raise ValueError(
#                 f"The number of control points (inDim) must be an integer. Current : {inDim}"
#             )
#         self.m = inDim - 1
#         self.t = torch.arange(-self.k, self.m + self.k) / (self.m - 1)

#     def forward(self, x: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass method to compute batched B-spline interpolation.

#         Args:
#             x (torch.Tensor): Input tensor of shape (*,B, nEval).
#             cp (torch.Tensor): Control points tensor of shape (B, inDim).

#         Returns:
#             torch.Tensor: Evaluated B-spline interpolation of shape (B, nEval).
#         """
#         assert (
#             cp.size(-1) == self.m + 1
#         ), "Control points must have m+1 element in the last dimension"
#         assert cp.size(-2) == x.size(
#             -2
#         ), "Both control points and x batch sizes must equals"
#         # Pads the control points with k-1 last elements
#         paddedCp = torch.cat([cp, cp[..., -1:].expand(*((cp.ndim-1)*[-1]), self.k - 1)], dim=-1)  # (B, m+k)
#         # Gets the bin indices that contains x
#         leftRange = (self.t.unsqueeze(-2) > x.clamp(0, 1).unsqueeze(-1)).float().argmax(
#             -1
#         ) - 1
#         # Create batched indices slices : B times a corresponding slice(m-k, m+1)
#         slicesIndices = leftRange.unsqueeze(-1) + torch.arange(-self.k, 1).unsqueeze(-2)
#         # Retrieve from control points all the batched slices
#         d = paddedCp.gather(-1, slicesIndices.flatten(-2)).unflatten(
#             -1, (slicesIndices.shape[-2:])
#         )
#         # Proceed to optimized batched de Boor algorithm
#         for r in range(1, self.k + 1):
#             for j in range(self.k, r - 1, -1):
#                 alphas = (x - self.t[j + leftRange - self.k]) / (
#                     self.t[j + 1 + leftRange - r] - self.t[j + leftRange - self.k]
#                 )
#                 d[..., j] = (1.0 - alphas) * d[..., j - 1] + alphas * d[..., j]
#         return d[..., self.k]