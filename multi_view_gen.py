import numpy as np
import json
import random


def camera_pose_sample(obj_center: list, cam_json, radius: float = 4.0, nums: int = 5):
    # 提取所有摄像机的位置为 (n, 3) 矩阵
    positions = np.array([cam['position'] for cam in cam_json])

    # 计算每个姿态与目标中心的距离
    center_distances = np.linalg.norm(positions - np.array(obj_center), axis=-1)

    # 选取与目标中心距离大于 radius 的姿态索引
    valid_indices = np.where(center_distances > radius)[0]

    if len(valid_indices) == 0:
        return []  # 没有有效的姿态

    # 筛选有效的姿态
    filtered_positions = positions[valid_indices]
    filtered_indices = valid_indices

    # 计算所有有效姿态之间的距离矩阵
    distance_matrix = np.linalg.norm(
        filtered_positions[:, np.newaxis, :] - filtered_positions[np.newaxis, :, :],
        axis=-1
    )

    pose_store = []

    # 初始化一个布尔掩码来跟踪有效姿态
    is_valid = np.ones(len(filtered_indices), dtype=bool)

    for _ in range(nums):
        if len(filtered_indices) == 0:
            break

        if len(pose_store) == 0:
            # 第一个姿态随机选择
            selected_index = random.choice(filtered_indices)
        else:
            # 后续姿态从距离已选姿态均大于 radius 的候选姿态中选择
            valid_mask = is_valid.copy()
            for pose_index in pose_store:
                # 找到 pose_index 在 filtered_indices 中的实际索引
                actual_index = np.where(filtered_indices == pose_index)[0][0]
                valid_mask &= distance_matrix[actual_index] > radius

            candidates = filtered_indices[valid_mask]

            if len(candidates) == 0:
                break

            selected_index = random.choice(candidates)

        pose_store.append(selected_index)
        # 更新布尔掩码，将已选中的姿态标记为无效
        is_valid[filtered_indices == selected_index] = False

    return pose_store

# with open("lerf_ovs/teatime/output/teatime/cameras.json", 'r') as file:
#     cam_json = json.load(file)
#
# center = [2.2567, 1.1445, 1.3868]  # 物体中心坐标
#
# camera_pose = camera_pose_sample(center, r, num_poses, cam_json)
# update_cam_json = [cam_json[idx] for idx in camera_pose]
# with open("custom_view/teatime_3/cameras.json", 'w') as file:
#     json.dump(update_cam_json, file)


