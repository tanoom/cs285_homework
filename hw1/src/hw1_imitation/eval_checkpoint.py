"""Evaluate a saved checkpoint and print the reward.

Usage:
    uv run src/hw1_imitation/eval_checkpoint.py --checkpoint path/to/checkpoint.pkl
    
    # Or download from wandb first:
    # wandb artifact get your-username/hw1-imitation-fm/policy-checkpoint-xxx:v0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import BasePolicy

ENV_ID = "gym_pusht/PushT-v0"


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> BasePolicy:
    """Load a trained policy from checkpoint."""
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return model


def evaluate_checkpoint(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    num_episodes: int = 100,
    chunk_size: int = 8,
    flow_num_steps: int = 10,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a policy checkpoint and return metrics.
    
    Returns:
        Dictionary with mean_reward, std_reward, min_reward, max_reward
    """
    model.eval()
    rewards: list[float] = []

    env = gym.make(ENV_ID, obs_type="state", render_mode=None)
    action_low = env.action_space.low
    action_high = env.action_space.high

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        max_reward = 0.0

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                with torch.no_grad():
                    pred_chunk = (
                        model.sample_actions(
                            state.unsqueeze(0), num_steps=flow_num_steps
                        )
                        .cpu()
                        .numpy()[0]
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, info = env.step(
                action.astype(np.float32)
            )
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1

        rewards.append(max_reward)
        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"Episode {ep_idx + 1}/{num_episodes}: max_reward = {max_reward:.4f}")

    env.close()
    
    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint .pkl file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (for normalizer)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Action chunk size",
    )
    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=10,
        help="Number of flow denoising steps",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load normalizer from data
    print("Loading data for normalizer...")
    zarr_path = download_pusht(args.data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device)
    model = model.to(device)

    # Evaluate
    print(f"\nEvaluating over {args.num_episodes} episodes...")
    results = evaluate_checkpoint(
        model=model,
        normalizer=normalizer,
        device=device,
        num_episodes=args.num_episodes,
        chunk_size=args.chunk_size,
        flow_num_steps=args.flow_num_steps,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Reward:  {results['mean_reward']:.4f}")
    print(f"Std Reward:   {results['std_reward']:.4f}")
    print(f"Min Reward:   {results['min_reward']:.4f}")
    print(f"Max Reward:   {results['max_reward']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
