"""Visualize the 2D joint distribution of action chunks."""

from __future__ import annotations

from pathlib import Path
import argparse

import gym_pusht  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from hw1_imitation.data import download_pusht, load_pusht_zarr, Normalizer
from hw1_imitation.model import BasePolicy

ENV_ID = "gym_pusht/PushT-v0"


def visualize_action_distribution(
    actions: np.ndarray,
    title: str = "2D Joint Distribution of Actions",
    save_path: Path | None = None,
) -> None:
    """Create a 2D heatmap of the action distribution.
    
    Args:
        actions: Array of shape (N, 2) containing action data.
        title: Title for the plot.
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Scatter plot
    ax = axes[0]
    ax.scatter(actions[:, 0], actions[:, 1], alpha=0.1, s=1)
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("Scatter Plot")
    ax.set_aspect("equal")
    
    # 2. Hexbin heatmap
    ax = axes[1]
    hb = ax.hexbin(actions[:, 0], actions[:, 1], gridsize=50, cmap="viridis", mincnt=1)
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("Hexbin Heatmap")
    ax.set_aspect("equal")
    plt.colorbar(hb, ax=ax, label="Count")
    
    # 3. 2D Histogram
    ax = axes[2]
    h, xedges, yedges, im = ax.hist2d(
        actions[:, 0], actions[:, 1], bins=50, cmap="viridis"
    )
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("2D Histogram")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Count")
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.show()


def visualize_per_dimension_histogram(
    actions: np.ndarray,
    title: str = "Per-Dimension Action Distribution",
    save_path: Path | None = None,
) -> None:
    """Create histograms for each action dimension.
    
    Args:
        actions: Array of shape (N, 2) containing action data.
        title: Title for the plot.
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for dim in range(2):
        ax = axes[dim]
        ax.hist(actions[:, dim], bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel(f"Action Dim {dim}")
        ax.set_ylabel("Count")
        ax.set_title(f"Dimension {dim} ({'x' if dim == 0 else 'y'})")
        
        # Add statistics
        mean = actions[:, dim].mean()
        std = actions[:, dim].std()
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.axvline(mean + std, color="orange", linestyle=":", label=f"Std: {std:.2f}")
        ax.axvline(mean - std, color="orange", linestyle=":")
        ax.legend()
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    plt.show()


def visualize_expert_vs_policy(
    expert_actions: np.ndarray,
    policy_actions: np.ndarray,
    title: str = "Expert vs Policy Action Distribution",
    save_path: Path | None = None,
) -> None:
    """Compare expert and policy action distributions with side-by-side plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    hb = ax.hexbin(
        expert_actions[:, 0],
        expert_actions[:, 1],
        gridsize=50,
        cmap="Blues",
        mincnt=1,
    )
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("Expert Actions")
    ax.set_aspect("equal")
    plt.colorbar(hb, ax=ax, label="Count")

    ax = axes[1]
    hb = ax.hexbin(
        policy_actions[:, 0],
        policy_actions[:, 1],
        gridsize=50,
        cmap="Reds",
        mincnt=1,
    )
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("Policy Actions")
    ax.set_aspect("equal")
    plt.colorbar(hb, ax=ax, label="Count")

    ax = axes[2]
    ax.scatter(
        expert_actions[:, 0],
        expert_actions[:, 1],
        alpha=0.05,
        s=1,
        label="Expert",
        color="blue",
    )
    ax.scatter(
        policy_actions[:, 0],
        policy_actions[:, 1],
        alpha=0.05,
        s=1,
        label="Policy",
        color="red",
    )
    ax.set_xlabel("Action Dim 0 (x)")
    ax.set_ylabel("Action Dim 1 (y)")
    ax.set_title("Overlay")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> BasePolicy:
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def collect_policy_actions(
    model: BasePolicy,
    normalizer: Normalizer,
    *,
    chunk_size: int,
    flow_num_steps: int,
    num_episodes: int,
) -> np.ndarray:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    actions: list[np.ndarray] = []

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state = (
                    torch.from_numpy(normalizer.normalize_state(obs))
                    .float()
                    .to(next(model.parameters()).device)
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
            obs, _, terminated, truncated, _ = env.step(action.astype(np.float32))
            actions.append(action)
            done = terminated or truncated
            chunk_index += 1

    env.close()
    return np.asarray(actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize expert and policy action distributions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset directory for Push-T data and normalizer.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a trained policy checkpoint to visualize rollouts.",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--flow-num-steps", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    # Download and load the dataset
    zarr_path = download_pusht(args.data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    
    print(f"Loaded {len(actions)} actions with shape {actions.shape}")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"Action mean: {actions.mean(axis=0)}")
    print(f"Action std: {actions.std(axis=0)}")
    
    # Create output directory for figures
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Visualize raw action distribution
    visualize_action_distribution(
        actions,
        title="2D Joint Distribution of Raw Actions",
        save_path=output_dir / "action_distribution_2d.png",
    )
    
    visualize_per_dimension_histogram(
        actions,
        title="Per-Dimension Distribution of Raw Actions",
        save_path=output_dir / "action_distribution_per_dim.png",
    )
    
    # Visualize normalized action distribution
    normalizer = Normalizer.from_data(states, actions)
    normalized_actions = normalizer.normalize_action(actions)
    
    visualize_action_distribution(
        normalized_actions,
        title="2D Joint Distribution of Normalized Actions",
        save_path=output_dir / "action_distribution_2d_normalized.png",
    )
    
    visualize_per_dimension_histogram(
        normalized_actions,
        title="Per-Dimension Distribution of Normalized Actions",
        save_path=output_dir / "action_distribution_per_dim_normalized.png",
    )

    if args.checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint from {args.checkpoint} on {device}...")
        model = load_checkpoint(args.checkpoint, device)
        policy_actions = collect_policy_actions(
            model,
            normalizer,
            chunk_size=args.chunk_size,
            flow_num_steps=args.flow_num_steps,
            num_episodes=args.num_episodes,
        )

        print(
            f"Collected {len(policy_actions)} policy actions with shape "
            f"{policy_actions.shape}"
        )
        visualize_expert_vs_policy(
            actions,
            policy_actions,
            title="Expert vs Policy Action Distribution (Raw)",
            save_path=output_dir / "expert_vs_policy_actions.png",
        )

        normalized_policy_actions = normalizer.normalize_action(policy_actions)
        visualize_expert_vs_policy(
            normalized_actions,
            normalized_policy_actions,
            title="Expert vs Policy Action Distribution (Normalized)",
            save_path=output_dir / "expert_vs_policy_actions_normalized.png",
        )


if __name__ == "__main__":
    main()
