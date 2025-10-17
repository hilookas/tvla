import tvla.models.gemma # Register ours
import tvla.models.paligemma # Register ours

from tvla.models.pi0.modeling_pi0 import Pi0Model
from tvla.models.pi0.configuration_pi0 import Pi0Config

model_id = "/home/ubuntu/tvla/pi05_libero_finetuned_transformers"

config = Pi0Config.from_pretrained(model_id)

model = Pi0Model.from_pretrained(
    model_id,
    # dtype=torch.bfloat16,
    # device_map="auto",
)

from lerobot.processor import PolicyProcessorPipeline

from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

import tvla.models.pi05.processor_pi05

preprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path="lerobot/pi05_libero_finetuned",
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": "cpu"}},
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path="lerobot/pi05_libero_finetuned",
    config_filename="policy_postprocessor.json",
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
    libero_env = LiberoEnv(task_suite_name, task_id, uea_repr=False)

    # Start episodes for current task
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        logging.info(f"Starting episode {task_episodes+1}...")

        obs = libero_env.reset(episode_idx)

        from collections import deque
        n_action_steps = 50
        action_queue = deque(maxlen=n_action_steps)

        try:
            while True:
                observation = {
                    'observation.state': torch.tensor(obs["original_state"]).unsqueeze(0),
                    'observation.images.image': torch.tensor(obs['image'][:, ::-1] / 255.0).permute(2, 0, 1).unsqueeze(0), # Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
                    'observation.images.image2': torch.tensor(obs['wrist_image'][:, ::-1] / 255.0).permute(2, 0, 1).unsqueeze(0),
                    "task": [obs["instruction"]],
                }

                observation_model = preprocessor(observation)
                with torch.inference_mode():
                    # Action queue logic for n_action_steps > 1
                    if len(action_queue) == 0:
                        state = None # for pi05
                        lang_tokens = observation_model["observation.language.tokens"]
                        lang_masks = observation_model["observation.language.attention_mask"]

                        images = []
                        img_masks = []

                        # Get device from model parameters
                        device = "cpu"

                        # Preprocess image features present in the batch
                        for key in ["observation.images.image", "observation.images.image2"]:
                            img = observation_model[key]

                            # Ensure tensor is on the same device as the model
                            if img.device != device:
                                img = img.to(device)

                            # Ensure float32 dtype for consistency
                            if img.dtype != torch.float32:
                                img = img.to(torch.float32)

                            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
                            assert img.shape[1] == 3  # Check if channels are in dimension 1

                            import torch.nn.functional as F  # noqa: N812

                            # Resize
                            img = F.interpolate(
                                img,
                                size=(224, 224),
                                mode="bilinear",
                                align_corners=False,
                            )

                            # Normalize from [0,1] to [-1,1] as expected by siglip
                            img = img * 2.0 - 1.0

                            images.append(img)
                            # Create mask (all ones for real images)
                            bsize = img.shape[0]
                            mask = torch.ones(bsize, dtype=torch.bool, device=device)
                            img_masks.append(mask)

                        # Create image features not present in the batch as fully 0 padded images
                        for _num_empty_cameras in range(len(["observation.images.empty_camera_0"])):
                            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
                            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
                            images.append(img)
                            img_masks.append(mask)

                        # images, img_masks, lang_tokens, lang_masks, state = preprocess_observation_pytorch(observation, train=False) # train=True when training
                        action = model.sample_action(images, img_masks, lang_tokens, lang_masks, state)
                        original_action_dim = 7

                        # Transpose to get shape (n_action_steps, batch_size, action_dim)
                        action_queue.extend(action[:, :, :original_action_dim].transpose(0, 1))

                    action_model = action_queue.popleft()

                action = postprocessor(action_model)

                action = action.cpu().numpy().squeeze(0)

                action = {
                    "original_action": action,
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
