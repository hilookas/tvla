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
from .transform import libero2uea_state, uea2libero_action, quat2axisangle

# Set numpy print options for better readability
np.set_printoptions(precision=6, suppress=True)

class LiberoEnv:
    """
    LIBERO environment wrapper for robotic manipulation tasks.

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO
    cd LIBERO
    pip install -e .
    pip install easydict==1.9 robosuite==1.4.0 bddl==1.0.1 future==0.18.2 matplotlib==3.5.3 # ref: requirements.txt

    Python 3.10 are verified with this environment.
    """

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

    def __init__(self, task_suite_name, task_id, log_path="logs/libero", uea_repr: bool=True):
        """Initialize LIBERO environment.

        Args:
            task_suite_name: Name of the task suite
            task_id: ID of the specific task
            log_path: Path to save logs and videos
        """
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.log_path = log_path
        self.uea_repr = uea_repr

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

        observation = {
            "image": image,              # RGB image from agent view
            "depth": depth,              # Depth map
            "depth_scale": depth_scale,  # Scale factor for depth
            "intrinsics": intrinsics,    # Camera intrinsic matrix
            "wrist_image": wrist_image,  # RGB image from wrist camera
            "instruction": self.task_description, # Task description
        }

        if self.uea_repr:
            Ttcp2cam, gripper_open = libero2uea_state(
                obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"],
                get_camera_extrinsic_matrix(self.env.sim, "agentview") # context
            )

            return observation | {
                "Ttcp2cam": Ttcp2cam,        # TCP pose in camera frame
                "gripper_open": gripper_open, # Gripper opening amount (in SI unit (meter))
            }
        else:
            return observation | {
                "original_state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
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

        if self.uea_repr:
            scaled_delta_pos, scaled_delta_ori, gripper_action = uea2libero_action(
                action["Ttcp2cam"], action["gripper_open"],
                self.last_obs["Ttcp2cam"], get_camera_extrinsic_matrix(self.env.sim, "agentview") # context
            )

            # Reconstruct libero action format
            original_action = np.concatenate([
                scaled_delta_pos,      # Position delta
                scaled_delta_ori,      # Orientation delta
                [gripper_action]       # Gripper action
            ])
        else:
            original_action = action["original_action"]

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
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
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
            # from tvla.data.utils import draw_xyz_axis
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