"""
Xinyuan Implementation of porting x2robot data to lerobot format.
"""

import os
import shutil
import tqdm 

import os
os.environ["LEROBOT_HOME"] = "/x2robot/brae/projects/.cache/lerobot"

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

print(LEROBOT_HOME)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import glob
import numpy as np

from x2robot_dataset.common.datasets.video_utils import decode_video_torchvision
from x2robot_dataset.common.data_preprocessing import process_action

REPO_NAME = "office-sort-and-fold-clothes"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATHS = [
    '/x2robot/zhengwei/10007/20250325-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250325-night-fold-item-towel-4',
    '/x2robot/zhengwei/10007/20250326-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250326-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250326-night-fold-item-towel-4',
    '/x2robot/zhengwei/10007/20250327-day-fold-item-towel-1',
    '/x2robot/zhengwei/10007/20250327-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250327-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250327-night-fold-item-towel-4',
    '/x2robot/zhengwei/10007/20250328-day-fold-item-towel-1',
    '/x2robot/zhengwei/10007/20250328-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250328-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250328-night-fold-item-towel-4',
    '/x2robot/zhengwei/10007/20250329-day-fold-item-towel-1',
    '/x2robot/zhengwei/10007/20250329-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250329-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250329-night-fold-item-towel-4',
    '/x2robot/zhengwei/10007/20250330-day-fold-item-towel',
    '/x2robot/zhengwei/10007/20250330-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250330-night-fold-item-towel-3',
    '/x2robot/zhengwei/10007/20250331-day-fold-item-towel',
    '/x2robot/zhengwei/10007/20250331-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250401-day-pick_up-item-towel-1',
    '/x2robot/zhengwei/10007/20250401-day-pick_up-item-towel-2',
    '/x2robot/zhengwei/10007/20250401-day-fold-item-towel-2',
    '/x2robot/zhengwei/10007/20250402-day-fold-item-towel-1',
]

FILE_CAME_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}

_ACTION_KEY_FULL_MAPPING = {
    'follow_right_arm_joint_pos': 'follow_right_joint_pos',
    'follow_right_arm_joint_dev': 'follow_right_joint_dev',
    'follow_right_arm_joint_cur': 'follow_right_joint_cur',
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'master_right_arm_joint_pos': 'master_right_joint_pos',
    'master_right_arm_joint_dev': 'master_right_joint_dev',
    'master_right_arm_joint_cur': 'master_right_joint_cur',
    'master_right_ee_cartesian_pos': 'master_right_position',
    'master_right_ee_rotation': 'master_right_rotation',
    'master_right_gripper': 'master_right_gripper',
    'follow_left_arm_joint_pos': 'follow_left_joint_pos',
    'follow_left_arm_joint_dev': 'follow_left_joint_dev',
    'follow_left_arm_joint_cur': 'follow_left_joint_cur',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper',
    'master_left_arm_joint_pos': 'master_left_joint_pos',
    'master_left_arm_joint_dev': 'master_left_joint_dev',
    'master_left_arm_joint_cur': 'master_left_joint_cur',
    'master_left_ee_cartesian_pos': 'master_left_position',
    'master_left_ee_rotation': 'master_left_rotation',
    'master_left_gripper': 'master_left_gripper',
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    'head_actions': 'head_rotation'

}
_ACTION_KEY_FULL_MAPPING_INV = {v:k for k,v in _ACTION_KEY_FULL_MAPPING.items()}

COPY_FROM_ZHENGWEI_DISK = True

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    print(f"COPY_FROM_ZHENGWEI_DISK: {COPY_FROM_ZHENGWEI_DISK}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ARX",
        fps=20,
        features={
            "face_view": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_view": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_view": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=12,
        image_writer_processes=6,
        copy_from_zhengwei_disk = COPY_FROM_ZHENGWEI_DISK,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_path in tqdm.tqdm(RAW_DATASET_PATHS):
        # raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        episode_paths = []

        for dataset_path in glob.glob(f'{raw_dataset_path}/*'):
            if os.path.isdir(dataset_path):
                has_mp4_files = len(glob.glob(f'{dataset_path}/*.mp4')) > 0
                if has_mp4_files:
                    episode_paths.append(dataset_path)
        
        # print(episode_paths): All the paths to the episodes (parent folders for each *.mp4)
        # assert False

        episode_num = 0
        for episode_path in tqdm.tqdm(episode_paths):
            
            ### process action
            pose_dicts = process_action(episode_path, action_key_mapping=_ACTION_KEY_FULL_MAPPING_INV)
            pose_dicts["follow_left_gripper"] = pose_dicts["follow_left_gripper"].reshape(-1, 1)
            pose_dicts["follow_right_gripper"] = pose_dicts["follow_right_gripper"].reshape(-1, 1)
            pose_dicts["master_left_gripper"] = pose_dicts["master_left_gripper"].reshape(-1, 1)
            pose_dicts["master_right_gripper"] = pose_dicts["master_right_gripper"].reshape(-1, 1)
            pred_poses = np.concatenate([pose_dicts[key] for key in ["follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper", "follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper"]], axis=1)
            real_poses = np.concatenate([pose_dicts[key] for key in ["follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper", "follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper"]], axis=1)
            # print(f"pred_poses: {pred_poses.shape}")
            # print(f"real_poses: {real_poses.shape}")
            # assert False
            # TODO: ADD trim stationary in the future

            ### process video
            video_frames = {}
            for key in ["face_view", "left_wrist_view", "right_wrist_view"]:
                if not dataset.copy_from_zhengwei_disk:
                    video_path = os.path.join(episode_path, FILE_CAME_MAPPING[key])
                    frames = decode_video_torchvision(video_path)
                    video_frames[key] = frames # uint8
                else:
                    video_path = os.path.join(episode_path, FILE_CAME_MAPPING[key])
                    video_frames[key] = [video_path] * len(pred_poses)

            for i in range(len(pred_poses)-1):
                dataset.add_frame(
                    {
                        "face_view": video_frames["face_view"][i],
                        "left_wrist_view": video_frames["left_wrist_view"][i],
                        "right_wrist_view": video_frames["right_wrist_view"][i],
                        "state": real_poses[i],
                        "actions": pred_poses[i+1],
                        # "task": 'do something fun', # temporary fix
                    }
                )
            dataset.save_episode(task=f'Sort and fold clothes') ### TODO 记得改！！！
            episode_num += 1
            # assert False
            # if episode_num >= 5:
            #     break


    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)
