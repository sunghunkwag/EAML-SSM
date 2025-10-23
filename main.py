# -*- coding: utf-8 -*-
"""
Main training and adaptation script for EAML-SSM.

This version is MODIFIED to:
1. Initialize an ExperienceBuffer.
2. Populate the buffer during data collection.
3. Pass the buffer to the Adapter for experience-augmented adaptation.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict
import gymnasium as gym

from core.ssm import StateSpaceModel
from meta_rl.meta_maml import MetaMAML
from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from env_runner.environment import Environment

# --- 1. Import the new ExperienceBuffer ---
from experience.experience_buffer import ExperienceBuffer


def collect_data(
    env,
    policy_model,
    experience_buffer: Optional[ExperienceBuffer],  # MODIFIED: Accept buffer
    num_episodes=10,
    max_steps_per_episode=100,
    device="cpu",
) -> Dict[str, torch.Tensor]:
    """
    Collects trajectory data and populates the experience buffer.
    """
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    policy_model.eval()

    obs = env.reset()
    hidden_state = policy_model.init_hidden(batch_size=env.batch_size)

    total_steps = 0
    for ep in range(num_episodes):
        steps_in_ep = 0
        done = False

        ep_obs_list = []
        ep_next_obs_list = []

        while not done and steps_in_ep < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action_logits, next_hidden_state = policy_model(obs_tensor, hidden_state)

                if isinstance(env.action_space, gym.spaces.Discrete):
                    n_actions = env.action_space.n
                    probs = torch.softmax(action_logits[:, :n_actions], dim=-1)
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = action_logits.cpu().numpy().flatten()

            next_obs, reward, done, info = env.step(action)

            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)

            # --- 2. Store data for buffer ---
            if experience_buffer is not None:
                ep_obs_list.append(obs_tensor.squeeze(0))
                # Assuming target is next_obs
                ep_next_obs_list.append(torch.tensor(next_obs, dtype=torch.float32).to(device))

            obs = next_obs
            hidden_state = next_hidden_state
            steps_in_ep += 1
            total_steps += 1

        # --- 3. Add episode data to buffer ---
        if experience_buffer is not None and len(ep_obs_list) > 0:
            obs_to_add = torch.stack(ep_obs_list)
            next_obs_to_add = torch.stack(ep_next_obs_list)
            experience_buffer.add(obs_to_add, next_obs_to_add)

        obs = env.reset()
        hidden_state = policy_model.init_hidden(batch_size=env.batch_size)

    return {
        "observations": torch.tensor(np.array(all_obs), dtype=torch.float32).unsqueeze(0).to(device),
        "actions": torch.tensor(np.array(all_actions), dtype=torch.long).unsqueeze(0).to(device),
        "rewards": torch.tensor(np.array(all_rewards), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device),
        "next_observations": torch.tensor(np.array(all_next_obs), dtype=torch.float32).unsqueeze(0).to(device),
    }


def train_meta(args, model, env, device, experience_buffer: ExperienceBuffer):  # MODIFIED
    """
    Meta-training with MetaMAML.
    Also populates the experience buffer.
    """
    print("Starting MetaMAML training...")
    meta_learner = MetaMAML(model=model, inner_lr=args.inner_lr, outer_lr=args.outer_lr)

    for epoch in range(args.num_epochs):
        data = collect_data(
            env,
            model,
            experience_buffer=experience_buffer,  # MODIFIED: Pass buffer to fill it
            num_episodes=args.episodes_per_task,
            max_steps_per_episode=100,
            device=device,
        )

        obs_seq = data["observations"]
        next_obs_seq = data["next_observations"]

        total_len = obs_seq.shape[1]
        if total_len < 2:
            print("Warning: Collected data is too short, skipping epoch.")
            continue

        split_idx = total_len // 2

        x_support = obs_seq[:, :split_idx]
        y_support = next_obs_seq[:, :split_idx]
        x_query = obs_seq[:, split_idx:]
        y_query = next_obs_seq[:, split_idx:]

        tasks = [(x_support, y_support, x_query, y_query)]
        initial_hidden = model.init_hidden(batch_size=1)

        loss = meta_learner.meta_update(tasks, initial_hidden_state=initial_hidden, loss_fn=nn.MSELoss())

        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            print(f"Epoch {epoch}, Meta Loss: {loss:.4f}, Buffer Size: {len(experience_buffer)}")

    print("MetaMAML training completed.")
    print(f"Experience Buffer populated with {len(experience_buffer)} experiences.")


def test_time_adapt(args, model, env, device, experience_buffer: ExperienceBuffer):  # MODIFIED
    """
    Test-time adaptation using the Experience-Augmented Adapter.
    """
    print(f"\nStarting test-time adaptation with Experience Buffer (Weight: {args.experience_weight})...")

    # Create adapter config
    config = AdaptationConfig(
        learning_rate=args.adapt_lr,
        num_steps=5,  # Internal steps per call
        experience_batch_size=args.experience_batch_size,
        experience_weight=args.experience_weight,
    )

    # --- 4. Pass the populated buffer to the Adapter ---
    adapter = Adapter(model=model, config=config, experience_buffer=experience_buffer, device=device)

    obs = env.reset()
    hidden_state = model.init_hidden(batch_size=1)

    for step in range(args.num_adapt_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        current_hidden_state_for_adapt = hidden_state

        with torch.no_grad():
            output, hidden_state = model(obs_tensor, current_hidden_state_for_adapt)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        next_obs, reward, done, info = env.step(action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Call the adapter. Its internal logic is now hybrid.
        loss_val, steps_taken = adapter.update_step(
            x_current=obs_tensor, y_current=next_obs_tensor, hidden_state_current=current_hidden_state_for_adapt
        )

        obs = next_obs

        if done:
            obs = env.reset()
            hidden_state = model.init_hidden(batch_size=1)

        if step % 10 == 0 or step == args.num_adapt_steps - 1:
            print(f"Adaptation step {step}, Hybrid Loss: {loss_val:.4f}, Steps taken: {steps_taken}")

    print("Adaptation completed.")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="EAML-SSM: Experience-Augmented Meta-RL")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Gymnasium environment name")
    parser.add_argument("--state_dim", type=int, default=32, help="SSM state dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="SSM hidden layer dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of meta-training epochs")
    parser.add_argument("--episodes_per_task", type=int, default=5, help="Episodes collected per meta-task")
    parser.add_argument("--batch_size", type=int, default=1, help="Environment batch size (only 1 supported)")
    parser.add_argument("--inner_lr", type=float, default=0.01, help="Inner learning rate for MetaMAML")
    parser.add_argument("--outer_lr", type=float, default=0.001, help="Outer learning rate for MetaMAML")
    parser.add_argument("--adapt_lr", type=float, default=0.01, help="Learning rate for test-time adaptation")
    parser.add_argument("--num_adapt_steps", type=int, default=50, help="Total number of adaptation steps during test")

    # --- 5. Add new args for the buffer ---
    parser.add_argument("--buffer_size", type=int, default=50000, help="Max size of the experience buffer")
    parser.add_argument(
        "--experience_batch_size", type=int, default=32, help="Batch size to sample from buffer during adaptation"
    )
    parser.add_argument("--experience_weight", type=float, default=0.1, help="Weight for the experience loss component")

    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: This example currently assumes batch_size=1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 6. Initialize the ExperienceBuffer ---
    experience_buffer = ExperienceBuffer(max_size=args.buffer_size, device=device)

    env = Environment(env_name=args.env_name, batch_size=args.batch_size)
    obs_space = env.observation_space

    input_dim = obs_space.shape[0] if isinstance(obs_space, gym.spaces.Box) else obs_space.n
    output_dim = input_dim  # Target is next_obs

    args.input_dim = input_dim
    args.output_dim = output_dim

    model = StateSpaceModel(
        state_dim=args.state_dim, input_dim=input_dim, output_dim=output_dim, hidden_dim=args.hidden_dim
    ).to(device)

    print(f"\n=== EAML-SSM (Hybrid Experience/Adaptation) ===")
    print(f"Environment: {args.env_name}, Device: {device}")
    print(f"Buffer Size: {args.buffer_size}, Experience Weight: {args.experience_weight}")
    print("=================================================\n")

    # Meta-Train (and populate buffer)
    train_meta(args, model, env, device, experience_buffer)

    # Test Time Adaptation (using populated buffer)
    test_time_adapt(args, model, env, device, experience_buffer)

    print("\n=== Hybrid Execution completed successfully ===")


if __name__ == "__main__":
    main()
