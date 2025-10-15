import numpy as np

from glob import glob
# from tvla.data.utils import polyfill_glob as glob # python 3.8

import argparse
from pathlib import Path
import shutil
import json
import h5py
from tqdm import tqdm

from libero.libero.envs import TASK_MAPPING

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map
from pytransform3d.transformations import invert_transform

from tvla.envs.libero.transform import libero2uea_state, libero2uea_action, axisangle2quat
from tvla.data.utils import VideoEncoder, MultiProcessVideoEncoder, NonDaemonPool

import logging

logger = None

def list_tasks(libero_base_path, success_only=True):
    task_hdf5_paths = glob("*/*.hdf5", root_dir=libero_base_path)

    logger.info(f"Found {len(task_hdf5_paths)} tasks")

    assert len(task_hdf5_paths) > 0

    return task_hdf5_paths

def rename_bddl_file(bddl_file_name, task_hdf5_path):
    # Rerender
    if "chiliocosm/bddl_files/" in bddl_file_name:
        bddl_file_name = bddl_file_name.replace("chiliocosm/bddl_files/", "libero/libero/libero/bddl_files/")

    if "libero_100_debug" in bddl_file_name:
        bddl_file_name = bddl_file_name.replace("libero_100_debug", "libero_100")

    if "libero_100" in bddl_file_name:
        if "libero_10" in task_hdf5_path:
            bddl_file_name = bddl_file_name.replace("libero_100", "libero_10")
        elif "libero_90" in task_hdf5_path:
            bddl_file_name = bddl_file_name.replace("libero_100", "libero_90")

    if "libero_spatial/pick_the_akita_" in bddl_file_name:
        bddl_file_name = bddl_file_name.replace("libero_spatial/pick_the_akita_", "libero_spatial/pick_up_the_")

    if "libero_object/pick_the_" in bddl_file_name:
        bddl_file_name = bddl_file_name.replace("libero_object/pick_the_", "libero_object/pick_up_the_")

    maptable = {
        "libero_10/STUDY_TABLETOP_SCENE1_pick_up_the_book_and_place_it_in_the_back_of_the_caddy.bddl": "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        "libero_90/KITCHEN_SCENE2_put_the_black_bowl_in_the_middle_on_the_plate.bddl": "libero_90/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl",
        "libero_90/KITCHEN_SCENE2_stack_the_black_bowl_in_the_middle_on_the_black_bowl_at_the_front.bddl": "libero_90/KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl",
        "libero_90/KITCHEN_TABLETOP_SCENE9_put_the_frypan_into_the_bottom_layer_of_the_cabinet.bddl": "libero_90/KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.bddl",
        "libero_90/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl",
        "libero_90/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.bddl",
        "libero_90/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.bddl",
        "libero_goal/open_the_middle_layer_of_the_drawer.bddl": "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
        "libero_goal/open_the_top_layer_of_the_drawer_and_put_the_bowl_inside.bddl": "libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl",
        "libero_goal/put_the_bowl_on_the_top_of_the_drawer.bddl": "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
        "libero_goal/put_the_cream_cheese_on_the_bowl.bddl": "libero_goal/put_the_cream_cheese_in_the_bowl.bddl",
        "libero_goal/put_the_wine_bottle_on_the_top_of_the_drawer.bddl": "libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.bddl",
        "libero_spatial/pick_up_the_black_bowl_in_the_top_layer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl": "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
        "libero_spatial/pick_up_the_black_bowl_next_to_the_cookies_box_and_place_it_on_the_plate.bddl": "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
        "libero_spatial/pick_up_the_black_bowl_on_the_cookies_box_and_place_it_on_the_plate.bddl": "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    }
    for k, v in maptable.items():
        if k in bddl_file_name:
            bddl_file_name = bddl_file_name.replace(k, v)
            break

    bddl_file_name = bddl_file_name.replace("libero/libero/libero/", "libero/libero/")

    return bddl_file_name

def process_one_task(args_tuple):
    """Process a single task with the given parameters."""
    task_hdf5_path, libero_base_path, output_path = args_tuple
    try:
        h5py_f = h5py.File(libero_base_path + "/" + task_hdf5_path, "r")

        for episode_idx in tqdm(range(h5py_f["data"].attrs["num_demos"])):
            episode_output_path = output_path + "/" + task_hdf5_path + "/" + str(episode_idx)
            Path(episode_output_path).mkdir(parents=True, exist_ok=True)

            # Get language instruction
            language_instruction = json.loads(h5py_f["data"].attrs["problem_info"])["language_instruction"]

            # Env meta
            env_args = json.loads(h5py_f["data"].attrs["env_args"])
            env_kwargs = env_args["env_kwargs"]

            fps = env_kwargs["control_freq"]
            assert fps == 20

            lowres_width = env_kwargs["camera_widths"]
            lowres_height = env_kwargs["camera_heights"]
            assert lowres_width == 128
            assert lowres_height == 128

            highres_width = 512
            highres_height = 512
            env_kwargs["camera_widths"] = highres_width
            env_kwargs["camera_heights"] = highres_height

            env_kwargs["camera_depths"] = True
            env_kwargs["camera_names"].append("sideview")

            libero_project_path = "/home/chq/tvla/LIBERO"
            env_kwargs["bddl_file_name"] = libero_project_path + "/" + rename_bddl_file(env_kwargs["bddl_file_name"], task_hdf5_path)

            # Save video in hdf5
            encoder_head = MultiProcessVideoEncoder(video_path=episode_output_path + "/" + "cam_head_lowres.mp4", fps=fps, width=lowres_width, height=lowres_height)
            encoder_hand = MultiProcessVideoEncoder(video_path=episode_output_path + "/" + "cam_hand_lowres.mp4", fps=fps, width=lowres_width, height=lowres_height)

            for img_head, img_hand in zip(h5py_f["data/demo_0/obs/agentview_rgb"], h5py_f["data/demo_0/obs/eye_in_hand_rgb"]):
                img_head = img_head[::-1] # Fix mujoco upside-down image
                img_hand = img_hand[::-1] # Fix mujoco upside-down image

                encoder_head.encode(img_head)
                encoder_hand.encode(img_hand)

            encoder_head.done()
            encoder_hand.done()

            # Rerender videos with high resolution

            # Make env
            env = TASK_MAPPING[env_args["problem_name"]](
                **env_kwargs,
            )

            obs = env.reset()

            # Camera params
            depth_scale = 4000

            intrinsics = get_camera_intrinsic_matrix(env.sim, "agentview", highres_height, highres_width)
            intrinsics_side = get_camera_intrinsic_matrix(env.sim, "sideview", highres_height, highres_width)
            intrinsics_hand = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", highres_height, highres_width) # hand camera has a 75 fovy

            assert np.allclose(intrinsics, intrinsics_side)

            Tworld2cam = invert_transform(get_camera_extrinsic_matrix(env.sim, "agentview"))
            Tworld2cam_side = invert_transform(get_camera_extrinsic_matrix(env.sim, "sideview"))

            # Get Tbase2cam
            body_name = "robot0_base"
            body_id = env.sim.model.body_name2id(body_name)
            Tbase2world = np.eye(4)
            Tbase2world[:3, 3] = env.sim.data.body_xpos[body_id]
            Tbase2world[:3, :3] = env.sim.data.body_xmat[body_id].reshape(3, 3)

            Tbase2cam = Tworld2cam @ Tbase2world
            Tbase2cam_side = Tworld2cam_side @ Tbase2world

            # Regen
            encoder_head = MultiProcessVideoEncoder(video_path=episode_output_path + "/" + "cam_head.mp4", fps=fps, width=highres_width, height=highres_height)
            encoder_hand = MultiProcessVideoEncoder(video_path=episode_output_path + "/" + "cam_hand.mp4", fps=fps, width=highres_width, height=highres_height)
            encoder_side = MultiProcessVideoEncoder(video_path=episode_output_path + "/" + "cam_side.mp4", fps=fps, width=highres_width, height=highres_height)
            encoder_head_depth = VideoEncoder(video_path=episode_output_path + "/" + "cam_head.depth.mkv", fps=fps, width=highres_width, height=highres_height, vcodec="ffv1", pix_fmt="gray16le", frame_format="gray16le")
            encoder_hand_depth = VideoEncoder(video_path=episode_output_path + "/" + "cam_hand.depth.mkv", fps=fps, width=highres_width, height=highres_height, vcodec="ffv1", pix_fmt="gray16le", frame_format="gray16le")
            encoder_side_depth = VideoEncoder(video_path=episode_output_path + "/" + "cam_side.depth.mkv", fps=fps, width=highres_width, height=highres_height, vcodec="ffv1", pix_fmt="gray16le", frame_format="gray16le")

            Ttcp2cams = []
            Ttcp2cam_actions = []
            Ttcp2cam_sides = []
            Ttcp2cam_side_actions = []
            gripper_opens = []
            gripper_open_actions = []

            for state, action, eef_state, gripper_state in zip(h5py_f["data/demo_0/states"], h5py_f["data/demo_0/actions"], h5py_f["data/demo_0/obs/ee_states"], h5py_f["data/demo_0/obs/gripper_states"]):
                env.sim.set_state_from_flattened(state)
                obs, reward, done, info = env.step(action)

                img_head = obs["agentview_image"]
                img_hand = obs["robot0_eye_in_hand_image"]
                img_side = obs["sideview_image"]

                depth_head = (get_real_depth_map(env.sim, obs["agentview_depth"]) * depth_scale).astype(np.uint16).squeeze()
                depth_hand = (get_real_depth_map(env.sim, obs["robot0_eye_in_hand_depth"]) * depth_scale).astype(np.uint16).squeeze()
                depth_side = (get_real_depth_map(env.sim, obs["sideview_depth"]) * depth_scale).astype(np.uint16).squeeze()

                img_head = img_head[::-1]
                img_hand = img_hand[::-1]
                img_side = img_side[::-1]
                depth_head = depth_head[::-1]
                depth_hand = depth_hand[::-1]
                depth_side = depth_side[::-1]

                encoder_head.encode(img_head)
                encoder_hand.encode(img_hand)
                encoder_side.encode(img_side)
                encoder_head_depth.encode(depth_head)
                encoder_hand_depth.encode(depth_hand)
                encoder_side_depth.encode(depth_side)

                Ttcp2cam, gripper_open = libero2uea_state(
                    eef_state[:3], axisangle2quat(eef_state[3:6]), gripper_state,
                    invert_transform(Tworld2cam)
                )

                Ttcp2cam_action, gripper_open_action = libero2uea_action(
                    action[:3], action[3:6], action[6],
                    Ttcp2cam, invert_transform(Tworld2cam)
                )

                Ttcp2cam_side = Tworld2cam_side @ invert_transform(Tworld2cam) @ Ttcp2cam
                Ttcp2cam_side_action = Tworld2cam_side @ invert_transform(Tworld2cam) @ Ttcp2cam_action

                Ttcp2cams.append(Ttcp2cam)
                Ttcp2cam_actions.append(Ttcp2cam_action)
                Ttcp2cam_sides.append(Ttcp2cam_side)
                Ttcp2cam_side_actions.append(Ttcp2cam_side_action)
                gripper_opens.append(gripper_open)
                gripper_open_actions.append(gripper_open_action)

            encoder_head.done()
            encoder_hand.done()
            encoder_side.done()
            encoder_head_depth.done()
            encoder_hand_depth.done()
            encoder_side_depth.done()

            with open(episode_output_path + "/" + "data.json", "w") as f:
                json.dump({
                    "language_instruction": language_instruction,

                    "intrinsics": intrinsics.tolist(),
                    "depth_scale": depth_scale,
                    "intrinsics_side": intrinsics_side.tolist(),
                    "depth_scale_side": depth_scale,
                    "intrinsics_hand": intrinsics_hand.tolist(),
                    "depth_scale_hand": depth_scale,

                    "Tbase2cam": Tbase2cam.tolist(),
                    "Tbase2cam_side": Tbase2cam_side.tolist(),
                    "Tbase2world": Tbase2world.tolist(),
                    "Tworld2cam": Tworld2cam.tolist(),
                    "Tworld2cam_side": Tworld2cam_side.tolist(),

                    "Ttcp2cams": np.array(Ttcp2cams).tolist(),
                    "Ttcp2cam_actions": np.array(Ttcp2cam_actions).tolist(),

                    "Ttcp2cam_sides": np.array(Ttcp2cam_sides).tolist(),
                    "Ttcp2cam_side_actions": np.array(Ttcp2cam_side_actions).tolist(),

                    "gripper_opens": np.array(gripper_opens).tolist(),
                    "gripper_open_actions": np.array(gripper_open_actions).tolist(),
                }, f)

        h5py_f.close()
    except Exception as e:
        logger.error(f"{task_hdf5_path}: error")
        import traceback; logger.error(traceback.format_exc())
        logger.error(e)

def main(args):
    global logger
    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    logger = logging.getLogger(__name__)

    libero_base_path = args.libero_base_path
    output_path = args.output_path

    if Path(output_path).exists():
        if args.overwrite:
            shutil.rmtree(output_path)
        else:
            raise Exception(f"Output path {output_path} already exists")

    task_hdf5_paths = list_tasks(libero_base_path)

    # for task_hdf5_path in tqdm(task_hdf5_paths):
    #     process_one_task((task_hdf5_path, libero_base_path, output_path))

    with NonDaemonPool(args.num_workers) as pool:
        _ = list(tqdm(pool.imap(process_one_task, [
            (task_hdf5_path, libero_base_path, output_path)
            for task_hdf5_path in task_hdf5_paths
        ]), total=len(task_hdf5_paths)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_base_path", type=str, default="/mnt/20T/chq_large/libero_dataset/")
    parser.add_argument("--output_path", type=str, default="/mnt/18T/chq_large/tvla/libero")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()
    main(args)

# python src/tvla/data/libero/convert.py --logfile libero.log --overwrite --num_workers 4