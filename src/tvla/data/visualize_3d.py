import json
import argparse
from tvla.data.utils import VideoEncoder, draw_xyz_axis, load_video
import numpy as np
from tqdm import tqdm

import open3d as o3d
import numpy as np
import cv2

import pytransform3d.transformations as pt

# Copy from: https://github.com/Jianghanxiao/Helper3D/blob/master/trimesh_render/src/camera.py

# lookAt function implementation
def lookAt(eye, target, up, yz_flip=False):
    # Normalize the up vector
    up /= np.linalg.norm(up)
    forward = eye - target
    forward /= np.linalg.norm(forward)
    if np.dot(forward, up) == 1 or np.dot(forward, up) == -1:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    rotation = np.eye(4)
    rotation[:3, :3] = np.row_stack((right, new_up, forward))

    # Apply a translation to the camera position
    translation = np.eye(4)
    translation[:3, 3] = [
        np.dot(right, eye),
        np.dot(new_up, eye),
        -np.dot(forward, eye),
    ]

    if yz_flip:
        # This is for different camera setting, like Open3D
        rotation[1, :] *= -1
        rotation[2, :] *= -1
        translation[1, 3] *= -1
        translation[2, 3] *= -1

    camera_pose = np.linalg.inv(np.matmul(translation, rotation))

    return camera_pose

class PointCloudViewer:
    def __init__(self, video_save_path=None, fps=60):
        self.video_save_path = video_save_path
        if self.video_save_path is not None:
            # Initialize the pointcloud viewer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Point Cloud", width=1280, height=720)
        else:
            # Initialize the pointcloud viewer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Point Cloud") # full screen

        self.pcd = o3d.geometry.PointCloud()
        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.cord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.cord1_T_inv = np.eye(4)
        self.cord2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.cord2_T_inv = np.eye(4)

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.origin)
        self.vis.add_geometry(self.cord1)
        self.vis.add_geometry(self.cord2)

        render_option = self.vis.get_render_option()
        render_option.point_size = 1
        # render_option.background_color = np.asarray([0, 0, 0])

        view_control = self.vis.get_view_control()
        view_control.set_constant_z_near(0)
        view_control.set_constant_z_far(1000)

        # Retrieve the camera parameters
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Set the extrinsic parameters, yz_flip is for Open3D camera configuration
        # camera_pose = lookAt(eye=np.array([0., 0., -1.]), target=np.array([0. ,0., 0.]), up=np.array([0.0, -1.0, 0.0]), yz_flip=True)
        camera_pose = lookAt(eye=np.array([0., -1., -1.]), target=np.array([0. ,0., 0.]), up=np.array([0.0, -1.0, 0.0]), yz_flip=True)
        camera_params.extrinsic = np.linalg.inv(camera_pose)

        # Set the camera parameters
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        if self.video_save_path is not None:
            frame = self.vis.capture_screen_float_buffer(do_render=True)
            frame = np.asarray(frame)
            self.video_writer = cv2.VideoWriter(self.video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

    def update_vis(self):
        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

    def update(self, image_array_rgb, depth_array, intrinsics, depth_scale, Tcord12cam=None, Tcord22cam=None):
        im_rgb = o3d.geometry.Image(image_array_rgb)
        im_d = o3d.geometry.Image(depth_array)

        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_rgb,
            im_d,
            depth_scale=float(depth_scale),
            convert_rgb_to_intensity=False
        )

        height, width, _ = image_array_rgb.shape

        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, o3d.camera.PinholeCameraIntrinsic(width=width, height=height, intrinsic_matrix=intrinsics))
        # new_pcd.transform(CAMERA_TO_WORLD)

        self.pcd.points = new_pcd.points
        self.pcd.colors = new_pcd.colors
        self.vis.update_geometry(self.pcd)

        if Tcord12cam is not None:
            self.cord1.transform(self.cord1_T_inv)
            self.cord1.transform(Tcord12cam)
            self.cord1_T_inv = pt.invert_transform(Tcord12cam)
            self.vis.update_geometry(self.cord1)

        if Tcord22cam is not None:
            self.cord2.transform(self.cord2_T_inv)
            self.cord2.transform(Tcord22cam)
            self.cord2_T_inv = pt.invert_transform(Tcord22cam)
            self.vis.update_geometry(self.cord2)

        self.update_vis()

        if self.video_save_path is not None:
            # Capture the current frame
            frame = self.vis.capture_screen_float_buffer(do_render=True)
            frame = (np.asarray(frame) * 255).astype(np.uint8)

            # Write frame to video
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.video_save_path is not None:
            self.video_writer.release()

        self.vis.destroy_window()

# viewer = PointCloudViewer()

def main(args):
    episode_path = args.episode_path
    cam_type = args.cam_type
    save_path = args.save_path

    with open(episode_path + "/" + "data.json", "r") as f:
        data = json.load(f)

    if cam_type == "head":
        video_name = "cam_head.mp4"
        depth_video_name = "cam_head.depth.mkv"
        intrinsics = np.array(data["intrinsics"])
        depth_scale = np.array(data["depth_scale"])
        if args.show_action:
            if "Ttcp2cam_actions" not in data or data["Ttcp2cam_actions"] is None:
                raise ValueError("Ttcp2cam_actions not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_actions"])
        else:
            if "Ttcp2cams" not in data or data["Ttcp2cams"] is None:
                raise ValueError("Ttcp2cams not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cams"])
    elif cam_type == "side":
        video_name = "cam_side.mp4"
        depth_video_name = "cam_side.depth.mkv"
        intrinsics = np.array(data["intrinsics_side"])
        depth_scale = np.array(data["depth_scale_side"])
        if args.show_action:
            if "Ttcp2cam_side_actions" not in data or data["Ttcp2cam_side_actions"] is None:
                raise ValueError("Ttcp2cam_side_actions not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_side_actions"])
        else:
            if "Ttcp2cam_sides" not in data or data["Ttcp2cam_sides"] is None:
                raise ValueError("Ttcp2cam_sides not found in data or is None")
            Ttcp2cams = np.array(data["Ttcp2cam_sides"])

    gripper_opens = np.array(data["gripper_opens"])

    viewer = PointCloudViewer(video_save_path=save_path, fps=args.fps)

    for frame, depth_frame, Ttcp2cam, gripper_open in zip(
        load_video(episode_path + "/" + video_name),
        load_video(episode_path + "/" + depth_video_name, format="gray16le"),
        tqdm(Ttcp2cams),
        gripper_opens
    ):
        if args.show_finger:
            Tleftfinger2tcp = np.eye(4)
            Tleftfinger2tcp[1, 3] = gripper_open / 2
            Tleftfinger2cam = Ttcp2cam @ Tleftfinger2tcp
            Trightfinger2tcp = np.eye(4)
            Trightfinger2tcp[1, 3] = -gripper_open / 2
            Trightfinger2cam = Ttcp2cam @ Trightfinger2tcp
            viewer.update(frame, depth_frame, intrinsics, depth_scale, Tleftfinger2cam, Trightfinger2cam)
        else:
            viewer.update(frame, depth_frame, intrinsics, depth_scale, Ttcp2cam)

    viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_path", type=str)
    parser.add_argument("--cam_type", type=str, default="head")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--show_finger", action="store_true", default=False)
    parser.add_argument("--show_action", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

# export DISPLAY=:10.0
# export __GLX_VENDOR_LIBRARY_NAME=mesa  # See: https://superuser.com/questions/106056/force-software-based-opengl-rendering-on-ubuntu
# python src/tvla/data/visualize_3d.py /mnt/20T/chq_large/tvla/droid/IPRL/success/2023-06-27/Tue_Jun_27_20:25:39_2023 --show_finger --cam_type side --save_path video_side_finger_o3d.mp4
