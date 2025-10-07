# Ref: openpi:examples/libero/main.py

import dataclasses
import logging
import pathlib

import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map
from robosuite.utils.transform_utils import axisangle2quat, quat2mat, mat2quat, quat2axisangle
from robosuite.utils.control_utils import set_goal_orientation, set_goal_position
from pytransform3d.transformations import invert_transform

# Set numpy print options for better readability
np.set_printoptions(precision=6, suppress=True)
      
# Copy from FoundationPose/Utils.py
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

class LiberoEnv:
    """LIBERO environment wrapper for robotic manipulation tasks."""
    
    @classmethod
    def list_tasks(cls, select_task_suite: None|str|list[str]=None):
        """List available tasks from LIBERO benchmark suites."""
        # Determine which task suites to use
        if select_task_suite is None:
            task_suite_names = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
        elif isinstance(select_task_suite, str):
            task_suite_names = [select_task_suite]
        elif isinstance(select_task_suite, list):
            task_suite_names = select_task_suite
        else:
            raise ValueError(f"Invalid task suite name: {select_task_suite}")
        
        benchmark_dict = benchmark.get_benchmark_dict()
        
        tasks = []
        for task_suite_name in task_suite_names:
            task_suite: benchmark.Benchmark = benchmark_dict[task_suite_name]()
            for task_id in range(task_suite.n_tasks):
                tasks.append((task_suite_name, task_id))
        return tasks
    
    def __init__(self, task_suite_name, task_id, log_path="logs/libero"):
        """Initialize LIBERO environment.
        
        Args:
            task_suite_name: Name of the task suite
            task_id: ID of the specific task
            log_path: Path to save logs and videos
        """
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.log_path = log_path
        
        # Set random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite: benchmark.Benchmark = benchmark_dict[self.task_suite_name]()
        logging.info(f"Task suite: {self.task_suite_name}")

        # Set maximum steps based on task suite (based on longest training demos)
        if self.task_suite_name == "libero_spatial":
            self.max_steps = 220  # longest training demo has 193 steps
        elif self.task_suite_name == "libero_object":
            self.max_steps = 280  # longest training demo has 254 steps
        elif self.task_suite_name == "libero_goal":
            self.max_steps = 300  # longest training demo has 270 steps
        elif self.task_suite_name == "libero_10":
            self.max_steps = 520  # longest training demo has 505 steps
        elif self.task_suite_name == "libero_90":
            self.max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {self.task_suite_name}")
        
        # Get specific task and its initial states
        task = task_suite.get_task(self.task_id)
        self.initial_states = task_suite.get_task_init_states(self.task_id)

        # Initialize LIBERO environment and task description
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        resolution = 256  # resolution used to render training data
        env_args = {
            "bddl_file_name": task_bddl_file, 
            "camera_heights": resolution, 
            "camera_widths": resolution,
            "camera_depths": True,
        }
        self.env = OffScreenRenderEnv(**env_args)
        seed = 7
        self.env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        
        self.task_description = task.language
        
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
    
    def reset(self, episode_idx):
        """Reset environment for a new episode.
        
        Args:
            episode_idx: Index of the episode
            
        Returns:
            Initial observation
        """
        self.episode_idx = episode_idx
        logging.info(f"Task: {self.task_description}")

        # Reset environment
        self.env.reset()

        # Set initial states
        obs = self.env.set_init_state(self.initial_states[self.episode_idx])
        
        # Wait for objects to stabilize in simulation
        num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
        for _ in range(num_steps_wait):
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # No movement, gripper open
            obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION)
        
        self.last_obs = self.compute_observation(obs)
        self.last_obs["success"] = done

        # Setup
        self.t = 0
        self.replay_images = []
        
        return self.last_obs
        
    # Transformation matrix from TCP (Tool Center Point) to end-effector frame
    Ttcp2eef = np.array([
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    
    def compute_observation(self, obs):
        """Compute observation from raw environment observation.
        
        Args:
            obs: Raw observation from LIBERO environment
            
        Returns:
            Processed observation dictionary
        """
        # Get RGB image and fix mujoco upside-down image
        image = obs["agentview_image"][::-1]  # rgb24
        depth_scale = 4000
        # Get depth map and fix upside-down
        depth = (get_real_depth_map(self.env.sim, obs["agentview_depth"]) * depth_scale)[::-1].squeeze().astype(np.uint16)  # gray16
        # Get camera intrinsic matrix
        intrinsics = get_camera_intrinsic_matrix(self.env.sim, "agentview", *depth.shape)
        
        # Get wrist camera image (eye-in-hand)
        wrist_image = obs["robot0_eye_in_hand_image"][::-1]  # Fix mujoco upside-down image # rgb24
        
        # Get camera extrinsic matrix and its inverse
        Tcam2world = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        Tworld2cam = invert_transform(Tcam2world)

        # Construct end-effector pose in world frame
        Teef2world = np.eye(4)
        Teef2world[:3, 3] = obs["robot0_eef_pos"]  # Position
        Teef2world[:3, :3] = quat2mat(obs["robot0_eef_quat"])  # Orientation

        # Compute TCP pose in camera frame
        Ttcp2cam = Tworld2cam @ Teef2world @ self.Ttcp2eef
        
        # Compute gripper opening
        gripper_open = float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
        
        return {
            "image": image,              # RGB image from agent view
            "depth": depth,              # Depth map
            "depth_scale": depth_scale,  # Scale factor for depth
            "intrinsics": intrinsics,    # Camera intrinsic matrix
            "wrist_image": wrist_image,  # RGB image from wrist camera
            "Ttcp2cam": Ttcp2cam,        # TCP pose in camera frame
            "gripper_open": gripper_open, # Gripper opening amount (in SI unit (meter))
            "instruction": self.task_description, # Task description
        }
    
    def step(self, action):
        """Execute action in environment.
        
        Args:
            action: Action dictionary containing Ttcp2cam (absolute control) and gripper_open
            
        Returns:
            Updated observation
        """
        assert self.t < self.max_steps
        self.t += 1
        
        # Record last image for video replay
        self.replay_images.append(self.last_obs["image"])
        
        # Restore end-effector pose in world frame from last observation
        Tcam2world = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        Teef2world = Tcam2world @ self.last_obs["Ttcp2cam"] @ invert_transform(self.Ttcp2eef)
        
        # Compute target end-effector pose in world frame from action
        Teefaction2world = Tcam2world @ action["Ttcp2cam"] @ invert_transform(self.Ttcp2eef)  # Absolute control
        
        # Compute position delta in libero way
        delta_pos = Teefaction2world[:3, 3] - Teef2world[:3, 3]
        
        # Compute orientation delta in libero way
        # action_ori = rotation_mat_error @ obs_eef_ori_mat
        # So rotation_mat_error = action_ori @ obs_eef_ori_mat.T
        rotation_mat_error = Teefaction2world[:3, :3] @ Teef2world[:3, :3].T
        delta_quat = mat2quat(rotation_mat_error)
        delta_axis_angle = quat2axisangle(delta_quat)
        
        # Restore scaling factors in libero way
        scaled_delta_pos = delta_pos * 20
        scaled_delta_ori = delta_axis_angle * 2
        
        # Restore gripper action
        # action_gripper_qpos = (-gripper_action + 1) * 0.04
        # So gripper_action = 1 - action_gripper_qpos / 0.04
        gripper_action = 1 - action["gripper_open"] / 0.04
        
        # Reconstruct libero action format
        original_action = np.concatenate([
            scaled_delta_pos,      # Position delta
            scaled_delta_ori,      # Orientation delta
            [gripper_action]       # Gripper action
        ])
        
        # Execute action in environment
        obs, reward, done, info = self.env.step(original_action)
        
        # Update observation and success status
        self.last_obs = self.compute_observation(obs)
        self.last_obs["success"] = done
        
        return self.last_obs

    def finish(self):
        """Save replay video of the episode."""
        # Save a replay video of the episode
        suffix = "success" if self.last_obs["success"] else "failure"
        task_segment = self.task_description.replace(" ", "_")
        imageio.mimwrite(
            pathlib.Path(self.log_path) / f"rollout_{self.task_suite_name}_{self.task_id:02d}_{task_segment}_{suffix}_{self.episode_idx:02d}.mp4",
            [np.asarray(x) for x in self.replay_images],
            fps=10,
        )

@dataclasses.dataclass
class Args:
    """Command line arguments for LIBERO evaluation."""
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_trials_per_task: int = 10  # Number of rollouts per task # max: 100

def test_libero(args: Args) -> None:
    """Evaluate LIBERO tasks.
    
    Args:
        args: Command line arguments
    """
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_suite_name, task_id in tqdm.tqdm(LiberoEnv.list_tasks(args.task_suite_name)):
        libero_env = LiberoEnv(task_suite_name, task_id)

        # Start episodes for current task
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"Starting episode {task_episodes+1}...")
            
            obs = libero_env.reset(episode_idx)
            
            # Debug: uncomment to visualize TCP pose
            # import ipdb; ipdb.set_trace()
            # draw_xyz_axis(obs["image"], obs["Ttcp2cam"], obs["intrinsics"], scale=0.03, thickness=4, transparency=0, save_path="cv_image.png")
            
            try:
                # Mock action (simple test action)
                action = {
                    "Ttcp2cam": obs["Ttcp2cam"].copy(),
                    "gripper_open": 0,
                }
                # Move TCP slightly in x-direction
                action["Ttcp2cam"][0,3] -= 0.1 
                
                while True:
                    # Execute action in environment
                    obs = libero_env.step(action)
                    if obs["success"]:
                        task_successes += 1
                        total_successes += 1
                        break

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback; traceback.print_exc()

            # Update episode counters
            task_episodes += 1
            total_episodes += 1
            
            # Save video and log results
            libero_env.finish()  # save video

            # Log current results
            logging.info(f"Success: {obs['success']}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results for current task
        logging.info(f"Current task success rate: {task_successes} / {task_episodes} ({float(task_successes) / float(task_episodes)}%)")
        logging.info(f"Current total success rate: {total_successes} / {total_episodes} ({float(total_successes) / float(total_episodes)}%)")

    # Log overall results
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    """Main entry point for LIBERO evaluation."""
    logging.basicConfig(level=logging.INFO)
    tyro.cli(test_libero)