import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(gaussians_path, sh_degree=3):
    plydata = PlyData.read(gaussians_path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # concatenate all features
    features = np.concatenate(
        [
            features_dc.reshape(xyz.shape[0], -1),
            features_extra.reshape(xyz.shape[0], -1),
        ],
        axis=1,
    )
    features = np.concatenate([opacities, features, scales, rots], axis=1)

    X = torch.tensor(xyz, dtype=torch.float32).contiguous()
    y = torch.tensor(features, dtype=torch.float32).contiguous()

    return X, y


def save_ply(X, y, path):
    # Restore the original order of the features
    xyz = X.numpy()
    normals = np.zeros_like(xyz)
    opacities = y[:, 0].unsqueeze(-1).numpy()
    rotation = y[:, -4:].numpy()
    scale = y[:, -7:-4].numpy()
    features = y[:, 1:-7]
    f_dc = features[:, :3].contiguous().numpy()
    f_rest = features[:, 3:].contiguous().numpy()

    os.makedirs(os.path.dirname(path), exist_ok=True)

    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append("f_dc_{}".format(i))
    for i in range(f_rest.shape[1]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(scale.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(rotation.shape[1]):
        l.append("rot_{}".format(i))

    dtype_full = [(attribute, "f4") for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
