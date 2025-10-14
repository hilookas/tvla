import pyzed.sl as sl
import numpy as np

from tvla.data.utils import VideoEncoder

import argparse
import pickle as pkl

def load_svo(svo_path, frame_max=None, frame_step=1):
    """
    Load SVO file from ZED SDK.

    ```bash
    # Download ZED SDK
    wget https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu24

    # Install python api
    python /usr/local/zed/get_python_api.py
    ```
    """

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)

    # Open the camera
    err = zed.open(init_params)
    assert err == sl.ERROR_CODE.SUCCESS, f"Error {err}: Failed to open SVO file"

    # Print camera info
    cam_info = zed.get_camera_information()
    fps = round(cam_info.camera_configuration.fps)
    height = round(cam_info.camera_configuration.resolution.height)
    width = round(cam_info.camera_configuration.resolution.width)
    # print("ZED Model                 : {0}".format(cam_info.camera_model))
    # print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    # print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version, cam_info.sensors_configuration.firmware_version))
    # print("ZED Camera Resolution     : {0}x{1}".format(width, height))
    # print("ZED Camera FPS            : {0}".format(fps))

    # Get intrinsics
    left_cam_calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx = left_cam_calibration_params.fx
    fy = left_cam_calibration_params.fy
    cx = left_cam_calibration_params.cx
    cy = left_cam_calibration_params.cy
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    depth_scale = 1000

    yield intrinsics, depth_scale, fps, height, width

    # Create image
    image = sl.Mat()
    depth = sl.Mat()
    # point_cloud = sl.Mat()

    # Read frames
    frame_idx = 0
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        if frame_idx % frame_step == 0:
            zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve right image
            # zed.retrieve_measure(depth_r, sl.MEASURE.DEPTH_RIGHT) # Retrieve right depth
            # zed.retrieve_measure(point_cloud_r, sl.MEASURE.XYZRGBA_RIGHT)

            # Convert to numpy array
            image_array = image.get_data().copy() # 不加copy好像会导致内存泄漏
            depth_array = depth.get_data().copy()
            # point_cloud_array = point_cloud.get_data().copy()

            a = yield frame_idx, image_array, depth_array

            if frame_max is not None and frame_idx == frame_max:
                break

        frame_idx += 1

    # Close the camera
    zed.close()

    return

def transcode_svo(input_path, output_path):
    g = load_svo(input_path)

    intrinsics, depth_scale, real_fps, height, width = next(g)

    depth_scale *= 4 # depth_scale 1000 -> 4000 for uint16 save

    assert height == 720
    assert width == 1280

    encoder = VideoEncoder(video_path=output_path, fps=60, width=1280, height=720)
    encoder_depth = VideoEncoder(video_path=output_path.replace(".mp4", ".depth.mkv"), fps=60, width=1280, height=720, vcodec="ffv1", pix_fmt="gray16le", frame_format="gray16le")

    frame_count = 0

    for frame_idx, image_array, depth_array in g:
        image_rgb = image_array[:, :, 2::-1] # BGRA2RGB # shape=(720, 1280, 3), dtype=uint8

        depth_array *= 4 # depth_scale 1000 -> 4000 for uint16 save

        depth = np.nan_to_num(np.clip(depth_array, 0, 65535), nan=0).astype(np.uint16) # shape=(720, 1280, 3), dtype=uint16

        frame_count += 1

        # import ipdb; ipdb.set_trace()
        # import cv2
        # cv2.imwrite("image.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("depth.png", depth)

        encoder.encode(image_rgb)
        encoder_depth.encode(depth)

    encoder.done()
    encoder_depth.done()

    return intrinsics, depth_scale, real_fps, frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    ret = transcode_svo(args.input_path, args.output_path)
    pkl.dump(ret, open(args.output_path + ".return.pkl", "wb"))
