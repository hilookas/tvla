import numpy as np

from glob import glob
# from tvla.data.utils import polyfill_glob as glob # python 3.8

import argparse
import multiprocessing as mp
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import h5py
import logging
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

logger = None

# Robotiq 2F-85
# https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf#page=103.05
Ttcp2eef = np.array([
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0.162],
    [0, 0, 0, 1]
])

def T_from_xyzrpy(xyzrpy):
    T = np.eye(4)
    T[:3,3] = xyzrpy[:3]
    T[:3,:3] = pr.matrix_from_euler(xyzrpy[3:], 0, 1, 2, True)
    return T

def xyzrpy_from_T(T):
    xyzrpy = np.zeros(6)
    xyzrpy[:3] = T[:3,3]
    xyzrpy[3:] = pr.euler_from_matrix(T[:3,:3], 0, 1, 2, True)
    return xyzrpy

def load_anno(anno_path):
    with open(anno_path + "/cam2base_extrinsics.json") as f:
        anno_cam2base_extrinsics = json.load(f)

    with open(anno_path + "/cam2base_extrinsic_superset.json") as f:
        anno_cam2base_extrinsic_superset = json.load(f)

    with open(anno_path + "/droid_language_annotations.json") as f:
        anno_language_annotations = json.load(f)

    return anno_cam2base_extrinsics, anno_cam2base_extrinsic_superset, anno_language_annotations

def list_episodes(droid_base_path, success_only=True):
    if success_only:
        episode_meta_paths = glob("*/success/*/*/metadata_*.json", root_dir=droid_base_path)
    else:
        episode_meta_paths = glob("*/*/*/*/metadata_*.json", root_dir=droid_base_path)

    logger.info(f"Found {len(episode_meta_paths)} episodes") # Success: 59740 episodes, All: 74896 episodes

    assert len(episode_meta_paths) > 0

    return episode_meta_paths

def wo_prefix(p):
    return "/".join(p.split("/")[3:])

import subprocess
import pickle as pkl
import sys
import os

# Zed seems have memory leak problem
# Run transcode_svo in a subprocess and get the return from a pickle file
def transcode_svo_subprocess(input_path, output_path):
    transcode_script = os.path.join(os.path.dirname(__file__), "transcode_svo.py")

    pkl_path = output_path + ".return.pkl"
    cmd = [
        sys.executable, transcode_script,
        "--input_path", input_path,
        "--output_path", output_path,
    ]
    subprocess.check_call(cmd)

    with open(pkl_path, "rb") as pf:
        ret = pkl.load(pf)
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    return ret

def process_one_episode(args_tuple):
    """Process a single episode with the given parameters."""
    episode_meta_path, droid_base_path, output_path, anno_path = args_tuple
    try:
        anno_cam2base_extrinsics, anno_cam2base_extrinsic_superset, anno_language_annotations = load_anno(anno_path)

        prefix_path = "/".join(episode_meta_path.split("/")[:-1])

        assert not Path(output_path + "/" + prefix_path).exists()
        Path(output_path + "/" + prefix_path).mkdir(parents=True, exist_ok=True)

        with open(droid_base_path + "/" + episode_meta_path, "r") as f:
            metadata = json.load(f)

        intrinsics, depth_scale, real_fps, frame_count = transcode_svo_subprocess( # TODO handle real_fps is not 60 and frame_count mismatch with actions
            droid_base_path + "/" + prefix_path + "/" + wo_prefix(metadata["ext1_svo_path"]),
            output_path + "/" + prefix_path + "/" + "cam_head.mp4",
        )
        intrinsics_side, depth_scale_side, real_fps_side, frame_count_side = transcode_svo_subprocess(
            droid_base_path + "/" + prefix_path + "/" + wo_prefix(metadata["ext2_svo_path"]),
            output_path + "/" + prefix_path + "/" + "cam_side.mp4",
        )
        intrinsics_hand, depth_scale_hand, real_fps_hand, frame_count_hand = transcode_svo_subprocess(
            droid_base_path + "/" + prefix_path + "/" + wo_prefix(metadata["wrist_svo_path"]),
            output_path + "/" + prefix_path + "/" + "cam_hand.mp4",
        )

        uuid = metadata["uuid"]

        # Get language instruction
        if uuid in anno_language_annotations:
            language_instruction = anno_language_annotations[uuid]['language_instruction1'] + " | " + anno_language_annotations[uuid]['language_instruction2'] + " | " + anno_language_annotations[uuid]['language_instruction3'] + " | " +  metadata["current_task"]
        else:
            language_instruction = metadata["current_task"]
            logger.warning(f"{episode_meta_path}: uuid {uuid} not in droid_language_annotations")

        # Retrieve camera extrinsics
        Tbase2cam = None
        Tbase2cam_side = None

        if uuid in anno_cam2base_extrinsic_superset:
            Tbase2cam = pt.invert_transform(T_from_xyzrpy(anno_cam2base_extrinsic_superset[uuid][metadata["ext1_cam_serial"]]))
            Tbase2cam_side = pt.invert_transform(T_from_xyzrpy(anno_cam2base_extrinsic_superset[uuid][metadata["ext2_cam_serial"]]))
        elif uuid in anno_cam2base_extrinsics:
            if metadata["ext1_cam_serial"] in anno_cam2base_extrinsics[uuid]:
                Tbase2cam = pt.invert_transform(T_from_xyzrpy(anno_cam2base_extrinsics[uuid][metadata["ext1_cam_serial"]]))
            if metadata["ext2_cam_serial"] in anno_cam2base_extrinsics[uuid]:
                Tbase2cam_side = pt.invert_transform(T_from_xyzrpy(anno_cam2base_extrinsics[uuid][metadata["ext2_cam_serial"]]))

        if Tbase2cam is None:
            logger.warning(f"{episode_meta_path}: Tbase2cam is None")
        if Tbase2cam_side is None:
            logger.warning(f"{episode_meta_path}: Tbase2cam_side is None")

        # Get Ttcp2base and gripper_open
        h5_path = prefix_path + "/" + wo_prefix(metadata["hdf5_path"])

        with h5py.File(droid_base_path + "/" + h5_path, "r") as f:
            # Gripper open
            griper_open_max = 0.085 # m
            gripper_opens = (1 - np.array(f["observation"]["robot_state"]["gripper_position"])) * griper_open_max # TODO gripper finger visualize
            gripper_open_actions = (1 - np.array(f["action"]['gripper_position'])) * griper_open_max

            # TCP
            cartesian_positions = np.array(f["observation"]["robot_state"]["cartesian_position"])
            cartesian_position_actions = np.array(f["action"]['cartesian_position'])

            assert len(gripper_opens) == len(gripper_open_actions) == len(cartesian_positions) == len(cartesian_position_actions)

            # State
            Ttcp2bases = np.array([T_from_xyzrpy(cartesian_position) @ Ttcp2eef for cartesian_position in cartesian_positions])
            if Tbase2cam is not None:
                Ttcp2cams = np.array([Tbase2cam @ Ttcp2base for Ttcp2base in Ttcp2bases])
            else:
                Ttcp2cams = None
            if Tbase2cam_side is not None:
                Ttcp2cam_sides = np.array([Tbase2cam_side @ Ttcp2base for Ttcp2base in Ttcp2bases])
            else:
                Ttcp2cam_sides = None

            # Action
            Ttcp2base_actions = np.array([T_from_xyzrpy(cartesian_position) @ Ttcp2eef for cartesian_position in cartesian_position_actions])
            if Tbase2cam is not None:
                Ttcp2cam_actions = np.array([Tbase2cam @ Ttcp2base for Ttcp2base in Ttcp2bases])
            else:
                Ttcp2cam_actions = None
            if Tbase2cam_side is not None:
                Ttcp2cam_side_actions = np.array([Tbase2cam_side @ Ttcp2base for Ttcp2base in Ttcp2bases])
            else:
                Ttcp2cam_side_actions = None

        with open(output_path + "/" + prefix_path + "/" + "data.json", "w") as f:
            json.dump({
                "language_instruction": language_instruction,

                "intrinsics": intrinsics.tolist(),
                "depth_scale": depth_scale,
                "intrinsics_side": intrinsics_side.tolist(),
                "depth_scale_side": depth_scale_side,
                "intrinsics_hand": intrinsics_hand.tolist(),
                "depth_scale_hand": depth_scale_hand,

                "Ttcp2base": Ttcp2bases.tolist(),
                "Ttcp2cam": Ttcp2cams.tolist() if Ttcp2cams is not None else None,
                "Ttcp2cam_side": Ttcp2cam_sides.tolist() if Ttcp2cam_sides is not None else None,

                "Ttcp2base_action": Ttcp2base_actions.tolist(),
                "Ttcp2cam_action": Ttcp2cam_actions.tolist() if Ttcp2cam_actions is not None else None,
                "Ttcp2cam_side_action": Ttcp2cam_side_actions.tolist() if Ttcp2cam_side_actions is not None else None,

                "gripper_open": gripper_opens.tolist(),
                "gripper_open_action": gripper_open_actions.tolist(),

                "Tbase2cam": Tbase2cam.tolist() if Tbase2cam is not None else None,
                "Tbase2cam_side": Tbase2cam_side.tolist() if Tbase2cam_side is not None else None,

                # aux info:
                "real_fps": real_fps,
                "frame_count": frame_count,
                "real_fps_side": real_fps_side,
                "frame_count_side": frame_count_side,
                "real_fps_hand": real_fps_hand,
                "frame_count_hand": frame_count_hand,
            }, f)
    except Exception as e:
        logger.error(f"{episode_meta_path}: error")
        import traceback; logger.error(traceback.format_exc())
        logger.error(e)


def main(args):
    global logger
    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    logger = logging.getLogger(__name__)

    droid_base_path = args.droid_base_path
    output_path = args.output_path
    anno_path = args.anno_path

    if Path(output_path).exists():
        if args.overwrite:
            shutil.rmtree(output_path)
        else:
            raise Exception(f"Output path {output_path} already exists")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    episodes = list_episodes(droid_base_path, success_only=not args.all)

    with mp.Pool(args.num_workers) as pool:
        r = list(tqdm(pool.imap(process_one_episode, [
            (episode, droid_base_path, output_path, anno_path)
            for episode in episodes
        ]), total=len(episodes)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", type=str, default="/home/chq/.cache/huggingface/hub/models--KarlP--droid/snapshots/bcb840c3b496533e0adf548a54b51f2f00057837")
    parser.add_argument("--droid_base_path", type=str, default="/mnt/20T/chq_large/droid_raw_1.0.1")
    parser.add_argument("--output_path", type=str, default="/mnt/20T/chq_large/tvla/droid")
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()
    main(args)

# python src/tvla/data/droid/convert.py --logfile droid.log --overwrite --num_workers 4