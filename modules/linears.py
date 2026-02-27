import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from train_utils.svd_smoother import train_svd_compressor, train_svd_smoother
import utils.low_rank_utils as lru


class SVDLinear(nn.Module):
    """ from https://github.com/hahnyuan/ASVD4LLM/blob/main/modules/svd_linear.py """
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)

        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        try:
            U, S, V = torch.svd_lowrank(w, q=rank)
        except:
            print(f"svd failed for {linear}, disable act_aware")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear

    def forward(self, inp):        
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y

    @staticmethod
    def smooth(x, lam=1):
        return torch.log1p(x)

    @staticmethod
    def inverse(y, lam=1):
        return torch.expm1(y)

class GradSVDLinear(nn.Module):
    """ from https://github.com/hahnyuan/ASVD4LLM/blob/main/modules/svd_linear.py """
    def __init__(self, weight, scale, bias, rank) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.rank = rank

    @staticmethod
    def from_linear(
        linear: nn.Linear, param_ratio: float, act_aware=False, ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV"
    ):
        if param_ratio >= 1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None
        return GradSVDLinear(w, scaling_diag_matrix, bias, rank)

    def forward(self, inp):
        w = self.weight * self.scale.view(1, -1)
        U, S, V = torch.svd_lowrank(w, q=self.rank)
        new_w = U.mul(S).mm(V.t())
        y = F.linear(inp, new_w, self.bias)
        return y
    
class LowRankLinear(nn.Module):
    """
    Replaces nn.Linear with a low-rank decomposition: W ≈ L @ R
    where L: (in_features, rank) and R: (rank, out_features)
    
    Supports both inference and training (e.g., for student-teacher training).
    """
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Low-rank factors
        self.L = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.R = nn.Parameter(torch.empty(rank, out_features, device=device, dtype=dtype))
        
        # Optional bias (usually present in transformer models)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize with proper scaling for training stability"""
        # Initialize L and R with Xavier-like initialization scaled by rank
        # This helps maintain activation scales during training
        fan_in = self.in_features
        fan_out = self.out_features
        
        # Scale initialization by 1/sqrt(rank) to keep output variance stable
        scale_L = math.sqrt(2.0 / (fan_in + self.rank))
        scale_R = math.sqrt(2.0 / (self.rank + fan_out))
        
        nn.init.normal_(self.L, std=scale_L)
        nn.init.normal_(self.R, std=scale_R)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute: x @ L @ R + bias
        # More efficient: x @ (L @ R) computed as (x @ L) @ R
        x_proj = torch.matmul(x, self.L)  # (..., rank)
        output = torch.matmul(x_proj, self.R)  # (..., out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, bias={self.bias is not None}')
        
        
class SVDLinear_Smoothed(nn.Module):
    """ from https://github.com/hahnyuan/ASVD4LLM/blob/main/modules/svd_linear.py """
    def __init__(self, L, R, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        self.ALinear = nn.Linear(L.size(1), L.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(R.size(1), R.size(0), bias=False)
        self.truncation_rank = L.size(0)
        
        self.ALinear.weight.data = L.contiguous()
        self.BLinear.weight.data = R.contiguous()
        
            
    @staticmethod
    def from_linear_with_trained_smoothing(linear: nn.Linear, param_ratio: float, rank_align=1, calib_data=None, args=None):
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        rank = compressed_params // (linear.in_features + linear.out_features)
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        
        L, R, loss_history = train_svd_compressor(linear.weight.data.float(), calib_data, rank, args, num_epochs=args.num_epochs, lr=1e-4)
        new_linear = SVDLinear_Smoothed(L, R)
        new_linear.to(linear.weight.dtype)
        return new_linear
        
        
        

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):

        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        try:
            U, S, V = torch.svd_lowrank(w, q=rank)
        except:
            print(f"svd failed for {linear}, disable act_aware")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y
