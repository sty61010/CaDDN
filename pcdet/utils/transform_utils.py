import torch
import kornia


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = kornia.convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # print("points.shape: ", points.shape)
    # print("points: ", points)

    # print("project.shape: ", project.shape)
    # print("project: ", project)

    # Transform points to image and get depths
    points_t = project @ points
    points_t = points_t.squeeze(dim=-1)

    # print("points_t.shape: ", points_t.shape)
    # print("points_t: ", points_t)

    points_img = kornia.convert_points_from_homogeneous(points_t)
    # print("points_img.shape: ", points_img.shape)
    # print("points_img: ", points_img)

    points_depth = points_t[..., -1] - project[..., 2, 3]
    # print("points_depth.shape: ", points_depth.shape)
    # print("points_depth: ", points_depth)

    return points_img, points_depth
