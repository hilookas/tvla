# DROID Dataset for T-VLA

## How to use

You need first download **raw** DROID dataset form <https://droid-dataset.github.io/droid/the-droid-dataset.html>

You also need download 3d annotated data from <https://huggingface.co/KarlP/droid>

```bash
hf download KarlP/droid
# The path of downloaded dataset will show in the last output of hf command.
```

## Convert to T-VLA UEA (Universal Explicit Action) Representation Format

```bash
python src/tvla/data/droid/convert.py --anno_path /home/chq/.cache/huggingface/hub/models--KarlP--droid/snapshots/bcb840c3b496533e0adf548a54b51f2f00057837 --droid_base_path /mnt/20T/chq_large/droid_raw_1.0.1 --output_path /mnt/18T/chq_large/tvla/droid --logfile droid.log --overwrite --num_workers 4
```

⚠️ Change paths according to your own data path.

## Visualize Converted Dataset

2D visualization using OpenCV:

```bash
python src/tvla/data/visualize_2d.py /mnt/18T/chq_large/tvla/droid/IPRL/success/2023-06-27/Tue_Jun_27_20:25:39_2023 --show_finger --cam_type side --show_action --save_path video_side_finger_action.mp4
```

⚠️ DROID have 3 cameras: ext1 ext2 wrist. We rename ext1 to cam_head as main camera, ext2 to cam_side, wrist to cam_hand.

use `--show_action` to visualize action instead of state trajectory.

use `--show_finger` to visualize finger instead of tool center point (TCP).

3D visualization using Open3D:

```bash
export DISPLAY=:10.0  # Change this base on your system. You can use `ssh -CXY` to forward X11 display to your local machine.
export __GLX_VENDOR_LIBRARY_NAME=mesa  # See: https://superuser.com/questions/106056/force-software-based-opengl-rendering-on-ubuntu
python src/tvla/data/visualize_3d.py /mnt/20T/chq_large/tvla/droid/IPRL/success/2023-06-27/Tue_Jun_27_20:25:39_2023 --show_finger --cam_type side --save_path video_side_finger_o3d.mp4
```