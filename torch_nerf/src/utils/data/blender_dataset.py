"""
blender_dataset.py - Abstraction of 'Blender' dataset.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data as data
from torch_nerf.src.utils.data.load_blender import load_blender_data


class NeRFBlenderDataset(data.Dataset):
    """
    Dataset object for loading 'synthetic blender' dataset.
    """

    def __init__(
        self,
        root_dir: str,
        scene_name: str,
        data_type: str,
        half_res: bool,
        white_bg: bool = True,
    ):
        """
        Constructor of 'NeRFBlenderDataset'.
        """
        # check arguments
        data_types = ["train", "val", "test"]
        if not data_type in data_types:
            raise ValueError(
                f"Unsupported dataset type. Expected one of {data_types}. Got {data_type}"
            )
        scene_names = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
        if not scene_name in scene_names:
            raise ValueError(
                f"Unsupported scene type. Expected one of {scene_names}. Got {scene_name}."
            )
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        assert root_dir.exists(), f"The directory {root_dir} does not exist."

        super().__init__()

        self._root_dir = root_dir / scene_name
        self._data_type = data_type
        self._white_bg = white_bg

        (
            self._imgs,
            self._poses,
            self._camera_params,
            self._render_poses,
            self._img_fnames,
        ) = load_blender_data(self._root_dir, self._data_type, half_res=half_res)

        self._img_height = self._camera_params[0]
        self._img_width = self._camera_params[1]
        self._focal_length = self._camera_params[2]

        # RGBA -> RGB
        if self._white_bg:
            self._imgs = self._imgs[..., :3] * self._imgs[..., -1:] + (1.0 - self._imgs[..., -1:])
        self._imgs = self._imgs[..., :3]

        # (4, 4) -> (3, 4)
        self._poses = self._poses[:, :3]
        self._render_poses = self._render_poses[:, :3]

        if self._imgs.shape[0] != self._poses.shape[0]:
            raise AssertionError(
                (
                    "Dataset sizes do not match. Got "
                    f"{self._imgs.shape[0]} images and {self._poses.shape[0]} camera poses.",
                )
            )

    def __len__(self) -> int:
        """Returns the total number of data in the dataset."""
        return self._imgs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data corresponding to the given index.

        Args:
            index (int): Index of the data to be retrieved.

        Returns:
            A tuple of torch.Tensor instances each representing input RGB images
                and camera extrinsic matrices.
        """
        img = torch.tensor(self._imgs[index])
        pose = torch.tensor(self._poses[index])

        return img, pose

    @property
    def img_height(self) -> int:
        return self._img_height

    @property
    def img_width(self) -> int:
        return self._img_width

    @property
    def focal_length(self) -> float:
        return self._focal_length

    @property
    def render_poses(self) -> torch.Tensor:
        return self._render_poses
