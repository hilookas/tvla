import numpy as np

import glob
import os

import av
from av.container import OutputContainer
from av.stream import Stream
from av.video.frame import VideoFrame
from fractions import Fraction
from pathlib import Path
import queue
import threading
import time

def polyfill_glob(pathname, *, root_dir):
    original_cwd = os.getcwd()
    try:
        os.chdir(root_dir)
        return glob.glob(pathname) # Pattern relative to the new CWD
    finally:
        os.chdir(original_cwd) # Always change back to the original CWD

class VideoEncoder:
    """
    Encode images to video using PyAV.

    ```bash
    # Install PyAV with libsvtav1 support
    git clone https://github.com/hilookas/PyAV.git -b v12.3.0-with-libsvtav1
    pushd PyAV

    sudo apt install -y python3-virtualenv libsvtav1enc-dev libx264-dev wget

    export PYAV_LIBRARY=ffmpeg-6.0
    source scripts/activate.sh
    ./scripts/build-deps
    # See: https://github.com/numpy/numpy/issues/22135
    # pip install "setuptools<=64.0.0"
    pip install -U "setuptools"
    make
    deactivate
    pip install -e .
    popd

    # (Optional)
    sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg6
    sudo apt install -y ffmpeg # upgrade ffmpeg to support libsvtav1
    ```

    See: https://github.com/hilookas/lerobot/blob/astra_robot_new/lerobot/common/datasets/video_utils.py
    """

    def __init__(
        self,
        video_path, # type: Path | str
        fps: float = 30,
        width: int = 1280,
        height: int = 720,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        frame_format: str = "rgb24",
        options = None, # type: dict[str, str] | None
        nice: int = 0,
    ):
        if isinstance(video_path, str):
            video_path = Path(video_path)

        video_path.parent.mkdir(parents=True, exist_ok=True)

        self.video_path: Path = video_path

        self.fps: float = fps
        self.width: int = width
        self.height: int = height
        self.vcodec: str = vcodec
        self.pix_fmt: str = pix_fmt
        self.frame_format: str = frame_format

        if options is None:
            options = {
                "g": str(2), # GOP Size,
                "crf": str(30),
                "preset": str(10),
            }

        self.options: dict[str, str] = options

        self.nice: int = nice

        self.q = queue.Queue()

        threading.Thread(target=self.thread, args=(), daemon=True).start()

    def thread(self):
        # ffmpeg -f image2 -r 30 -i imgs_dir/frame_%06d.png -vcodec libsvtav1 -pix_fmt yuv420p -g 2 -crf 30 -loglevel error -y imgs_dir.mp4

        container: OutputContainer = av.open(file=str(self.video_path), mode="w")

        stream: Stream = container.add_stream(self.vcodec, rate=self.fps, options=self.options)
        stream.pix_fmt = self.pix_fmt

        stream.width = self.width
        stream.height = self.height

        VIDEO_PTIME = 1 / self.fps
        VIDEO_CLOCK_RATE = 90000

        timestamp = 0

        while True:
            if self.nice:
                time.sleep(self.nice)

            image = self.q.get()

            if image is None:
                break
            # print(timestamp)

            frame = VideoFrame.from_ndarray(image, format=self.frame_format)

            frame.pts = timestamp
            frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)

            timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)

            for packet in stream.encode(frame):
                container.mux(packet)

            self.q.task_done()

        # Stop
        for packet in stream.encode(None):
            container.mux(packet)

        for stream in container.streams:
            stream.close()

        container.close()

        container = None

        self.q.task_done()

    def encode(self, image: np.ndarray):
        self.q.put(image)

    def done(self):
        self.q.put(None)
        self.q.join()

# Copy from https://github.com/NVlabs/FoundationPose/blob/main/Utils.py

def draw_xyz_axis(color, ob_in_cam, K=np.eye(3), scale=0.1, thickness=3, transparency=0, is_input_rgb=True, save_path=None):
    """
    Draw XYZ coordinate axes on an image.

    Args:
        color: Input image (RGB or BGR)
        ob_in_cam: Object pose in camera frame (4x4 transformation matrix)
        scale: Scale factor for axis length
        K: Camera intrinsic matrix (3x3)
        thickness: Line thickness for drawing
        transparency: Transparency factor (0-1)
        is_input_rgb: Whether input is RGB (True) or BGR (False)
        save_path: Optional path to save the result image

    Returns:
        Image with XYZ axes drawn
    """
    import cv2

    def project_3d_to_2d(pt, K, ob_in_cam):
        """Project 3D point to 2D image coordinates."""
        pt = pt.reshape(4, 1)
        projected = K @ ((ob_in_cam@pt)[:3,:])
        projected = projected.reshape(-1)
        projected = projected / projected[2]
        return projected.reshape(-1)[:2].round().astype(int)

    # Convert RGB to BGR if needed (OpenCV uses BGR)
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp = tmp.astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, tmp)

    # Convert back to RGB if input was RGB
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp