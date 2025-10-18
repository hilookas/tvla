import tvla.models.gemma # Register ours
import tvla.models.paligemma # Register ours

from tvla.models.pi0.modeling_pi0 import Pi0Model

from tvla.models.paligemma.processing_paligemma import PaliGemmaProcessor

from safetensors.torch import load_file as safe_load_file

from tvla.envs.libero.env import LiberoEnv, get_camera_extrinsic_matrix
from tvla.envs.libero.transform import uea2libero_state, libero2uea_action, quat2axisangle

import torch
import numpy as np

import tqdm
import logging

from collections import deque

model = Pi0Model.from_pretrained(
    "/home/ubuntu/tvla/pi05_libero_finetuned_transformers",
    # dtype=torch.bfloat16,
    # device_map="auto",
)

processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

state_normalizer_state_dict = safe_load_file("/home/ubuntu/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855/policy_preprocessor_step_2_normalizer_processor.safetensors")
action_unnormalizer_state_dict = safe_load_file("/home/ubuntu/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855/policy_postprocessor_step_0_unnormalizer_processor.safetensors")

state_norm_mean = state_normalizer_state_dict["observation.state.mean"].numpy()
state_norm_std = state_normalizer_state_dict["observation.state.std"].numpy()

action_norm_mean = action_unnormalizer_state_dict["action.mean"].numpy()
action_norm_std = action_unnormalizer_state_dict["action.std"].numpy()

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

        n_action_steps = 50
        action_queue = deque(maxlen=n_action_steps)

        try:
            while True:
                # Action queue logic for n_action_steps > 1
                if len(action_queue) == 0:
                    with torch.inference_mode():
                        state = None # for pi05

                        # State (Pi05)
                        state_np = (obs["original_state"] - state_norm_mean) / state_norm_std

                        max_state_dim = 32
                        padded_state_np = np.zeros(max_state_dim)
                        padded_state_np[:state_np.shape[0]] = state_np

                        cleaned_text = obs["instruction"].strip().replace("_", " ").replace("\n", " ")
                        state_str = " ".join(map(str, np.digitize(padded_state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1))
                        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

                        # Tokenizer of PaliGemma dont't add bos token which is different from default tokenizer
                        # Pi0 need default tokenizer which added bos token
                        #     from transformers import AutoTokenizer
                        #     AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")(full_prompt)
                        assert processor.tokenizer.add_bos_token == False
                        processor.tokenizer.add_bos_token = True
                        inputs = processor.tokenizer(
                            full_prompt,
                            max_length=200, # tokenizer_max_length
                            truncation=True,
                            padding="max_length",
                            padding_side="right",
                            return_tensors="pt",
                        )
                        processor.tokenizer.add_bos_token = False

                        lang_tokens = inputs["input_ids"]
                        lang_masks = inputs["attention_mask"].to(torch.bool)

                        # Image
                        pixel_values = processor.image_processor([
                            obs['image'][:, ::-1].transpose(2, 0, 1), # TODO
                            obs['wrist_image'][:, ::-1].transpose(2, 0, 1),
                            np.zeros((3, 256, 256)),
                        ])

                        images = [torch.tensor(pixel_value).unsqueeze(0) for pixel_value in pixel_values["pixel_values"]]
                        img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]

                        # images, img_masks, lang_tokens, lang_masks, state = preprocess_observation_pytorch(observation, train=False) # train=True when training
                        action = model.sample_action(images, img_masks, lang_tokens, lang_masks, state)

                        # Transpose to get shape (n_action_steps, action_dim)
                        original_action_dim = 7
                        action_queue.extend(action[:, :, :original_action_dim].squeeze(0).numpy())

                action_model = action_queue.popleft()

                action = action_model * action_norm_std + action_norm_mean

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
