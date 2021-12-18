# import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 3D convolution collapse module
        Args:
            channels [int]: Number of feature channels
            num_heights [int]: Number of height planes in voxel grid
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features [torch.Tensor(B, C, Z, Y, X)]: Voxel feature representation
        Returns:
            batch_dict:
                spatial_features [torch.Tensor(B, C, Y, X)]: BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        # print("voxel_features: ", voxel_features.shape)
        # print("bev_features: ", bev_features.shape)
        return batch_dict
