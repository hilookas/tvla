from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from lerobot.processor import PolicyProcessorPipeline

from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

policy = PI05Policy.from_pretrained("lerobot/pi05_libero_finetuned")

preprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path="lerobot/pi05_libero_finetuned",
    config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    overrides={"device_processor": {"device": str(policy.config.device)}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path="lerobot/pi05_libero_finetuned",
    config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    overrides={},
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)

from tvla.envs.libero.env import LiberoEnv, get_camera_extrinsic_matrix
from tvla.envs.libero.transform import uea2libero_state, libero2uea_action, quat2axisangle
import torch
import numpy as np

import tqdm
import logging

num_trials_per_task = 10

# Start evaluation
total_episodes, total_successes = 0, 0
for task_suite_name, task_id in tqdm.tqdm(LiberoEnv.list_tasks("libero_object")):
    libero_env = LiberoEnv(task_suite_name, task_id)

    # Start episodes for current task
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        logging.info(f"Starting episode {task_episodes+1}...")
        
        obs = libero_env.reset(episode_idx)
        
        try:
            while True:
                eef_pos, eef_quat, gripper_qpos = uea2libero_state(
                    obs["Ttcp2cam"], obs["gripper_open"],
                    get_camera_extrinsic_matrix(libero_env.env.sim, "agentview") # context
                )
                
                observation = {
                    'observation.state': torch.tensor(np.concatenate((
                        eef_pos,
                        quat2axisangle(eef_quat),
                        gripper_qpos,
                    ))).unsqueeze(0),
                    'observation.images.image': torch.tensor(obs['image'][:, ::-1] / 255.0).permute(2, 0, 1).unsqueeze(0), # Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
                    'observation.images.image2': torch.tensor(obs['wrist_image'][:, ::-1] / 255.0).permute(2, 0, 1).unsqueeze(0),
                    "task": [obs["instruction"]],
                }
                            
                observation = preprocessor(observation)
                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = postprocessor(action)
                
                action = action.cpu().numpy().squeeze(0)
                
                Ttcp2cam, gripper_open = libero2uea_action(
                    action[:3], action[3:6], action[6], 
                    obs["Ttcp2cam"], get_camera_extrinsic_matrix(libero_env.env.sim, "agentview") # context
                )
                
                action = {
                    "Ttcp2cam": Ttcp2cam,
                    "gripper_open": gripper_open,
                }
                
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
