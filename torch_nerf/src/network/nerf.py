"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.input_mlp = nn.Linear(pos_dim, feat_dim)
        self.mlp1 = nn.Linear(feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, feat_dim)
        self.mlp3 = nn.Linear(feat_dim, feat_dim)
        self.mlp4 = nn.Linear(feat_dim, feat_dim)

        self.mlp5 = nn.Linear(feat_dim+pos_dim, feat_dim) # layer after skip connection

        self.mlp6 = nn.Linear(feat_dim, feat_dim)
        self.mlp7 = nn.Linear(feat_dim, feat_dim)
        self.mlp8 = nn.Linear(feat_dim, feat_dim)

        self.sigma_head = nn.Linear(feat_dim, 1)

        self.mlp9 = nn.Linear(feat_dim+view_dir_dim, int(feat_dim/2)) #128 out_channels layer
        self.radiance_head = nn.Linear(int(feat_dim/2), 3) # radiance (RGB) head
        self.sigmoid = nn.Sigmoid()

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        fc = F.relu(self.input_mlp(pos))
        fc = F.relu(self.mlp1(fc))
        fc = F.relu(self.mlp2(fc))
        fc = F.relu(self.mlp3(fc))
        fc = F.relu(self.mlp4(fc))
        res1 = torch.cat((fc, pos), dim=1) # skip connection, dims: 256+gamma(pos_dim)=256+60=316

        fc = F.relu(self.mlp5(res1))
        fc = F.relu(self.mlp6(fc))
        fc = F.relu(self.mlp7(fc))
        
        sigma = F.relu(self.sigma_head(fc)) # sigma head. dims: 256 --> 1

        fc = self.mlp8(fc) # no activation (orange arrow layer)
        res2 = torch.cat((fc, view_dir), dim=1) # sigma feature vector dims: feature_dim+gamma(dir)=256+24=280  
        
        fc = F.relu(self.mlp9(res2))  # dims: 256+24=280 --> 128

        radiance = self.sigmoid(self.radiance_head(fc)) # radiance (RGB) head. dims: 128 --> 3
        return sigma,radiance

