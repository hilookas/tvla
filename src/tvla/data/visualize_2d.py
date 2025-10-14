import json
import argparse
from tvla.data.utils import VideoEncoder, draw_xyz_axis, load_video
import numpy as np
from tqdm import tqdm

def main(args):
    episode_path = args.episode_path
    cam_type = args.cam_type
    save_path = args.save_path

    with open(episode_path + "/" + "data.json", "r") as f:
        data = json.load(f)

    if cam_type == "head":
        video_name = "cam_head.mp4"
        intrinsics = np.array(data["intrinsics"])
        depth_scale = np.array(data["depth_scale"])
        if args.show_action:
            if "Ttcp2cam_action" not in data or data["Ttcp2cam_action"] is None:
                raise ValueError("Ttcp2cam_action not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_action"])
        else:
            if "Ttcp2cam" not in data or data["Ttcp2cam"] is None:
                raise ValueError("Ttcp2cam not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam"])
    elif cam_type == "side":
        video_name = "cam_side.mp4"
        intrinsics = np.array(data["intrinsics_side"])
        depth_scale = np.array(data["depth_scale_side"])
        if args.show_action:
            if "Ttcp2cam_side_action" not in data or data["Ttcp2cam_side_action"] is None:
                raise ValueError("Ttcp2cam_side_action not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_side_action"])
        else:
            if "Ttcp2cam_side" not in data or data["Ttcp2cam_side"] is None:
                raise ValueError("Ttcp2cam_side not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_side"])

    gripper_opens = np.array(data["gripper_open"])

    encoder = None
    for frame, Ttcp2cam, gripper_open in zip(load_video(episode_path + "/" + video_name), tqdm(Ttcp2cams), gripper_opens):
        if encoder is None:
            encoder = VideoEncoder(save_path, fps=args.fps, width=frame.shape[1], height=frame.shape[0])

        if args.show_finger:
            Tleftfinger2tcp = np.eye(4)
            Tleftfinger2tcp[1, 3] = gripper_open / 2
            Tleftfinger2cam = Ttcp2cam @ Tleftfinger2tcp
            frame = draw_xyz_axis(frame, Tleftfinger2cam, K=intrinsics, scale=0.05, thickness=3, transparency=0)
            Trightfinger2tcp = np.eye(4)
            Trightfinger2tcp[1, 3] = -gripper_open / 2
            Trightfinger2cam = Ttcp2cam @ Trightfinger2tcp
            frame = draw_xyz_axis(frame, Trightfinger2cam, K=intrinsics, scale=0.05, thickness=3, transparency=0)
        else:
            frame = draw_xyz_axis(frame, Ttcp2cam, K=intrinsics, scale=0.1, thickness=3, transparency=0)

        encoder.encode(frame)
    encoder.done()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_path", type=str)
    parser.add_argument("--cam_type", type=str, default="head")
    parser.add_argument("--save_path", type=str, default="video.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--show_finger", action="store_true", default=False)
    parser.add_argument("--show_action", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

# python src/tvla/data/visualize_2d.py /mnt/20T/chq_large/tvla/droid/IPRL/success/2023-06-27/Tue_Jun_27_20:25:39_2023 --show_finger --cam_type side --save_path video_side_finger.mp4