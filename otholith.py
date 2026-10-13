# import open3d as o3d
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn as nn
#
#
# def load_ply(file_path):
#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points)  # (N, 3)
#     return points
#
# def normalize_point_cloud(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     scale = np.max(np.linalg.norm(pc, axis=1))
#     pc = pc / scale
#     return pc
#
#
# class PointCloudDataset(Dataset):
#     def __init__(self, file_paths, labels):
#         self.file_paths = file_paths
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         pc = load_ply(self.file_paths[idx])
#         pc = normalize_point_cloud(pc)
#         pc = torch.tensor(pc, dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return pc, label
#
#
# class SimplePointNet(nn.Module):
#     def __init__(self, k=2):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool1d(1),  # global feature
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, k)
#         )
#
#     def forward(self, x):  # x: (B, N, 3)
#         x = self.fc(x.permute(0, 2, 1))  # (B, 256, N)
#         x = x.squeeze(-1)
#         return self.classifier(x)

import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationPlyDataset(Dataset):
    def __init__(self, ply_files, transforms=None):
        """
        ply_files: list of file paths
        transforms: optional function to augment data (rotate, jitter, etc.)
        """
        self.ply_files = ply_files
        self.transforms = transforms

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        file = self.ply_files[idx]
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points, dtype=np.float32)  # (N,3)

        # Try to load per-point labels if they exist
        labels = None
        if "label" in pcd.point_attr:  # hypothetical attribute
            labels = np.asarray(pcd.point_attr["label"], dtype=np.int64)
        elif pcd.colors is not None:
            # Example: map color to classes (if used)
            colors = np.asarray(pcd.colors, dtype=np.float32)
            labels = color_to_label(colors)  # define mapping

        else:
            raise ValueError(f"No labels found in {file}")

        if self.transforms:
            points, labels = self.transforms(points, labels)

        # Optionally normalize
        points = normalize_points(points)

        return torch.from_numpy(points), torch.from_numpy(labels)
