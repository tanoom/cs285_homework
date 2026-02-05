"""Interactive playground for Push-T policies with human perturbation.

This script allows you to:
- Load a trained policy and watch it execute
- Perturb the robot by clicking and dragging with your mouse
- Add forces/obstacles to challenge the policy
- Pause/resume execution to intervene

Controls:
- Left click + drag: Control red pusher OR drag agent (in sampling mode)
- S: Toggle sampling mode - drag agent to see multi-modal trajectories
- P: Toggle red pusher on/off (default: OFF)
- C: Clear sampled trajectories
- V: Toggle trajectory visualization during runtime
- SPACE: Pause/resume
- R: Reset episode
- M: Toggle manual mode (agent stops moving)
- T: Toggle unlimited time (ignore time limit)
- Q or ESC: Quit

Usage examples:
    # Manual mode only (no model):
    uv run python -m hw1_imitation.playground

    # With a trained checkpoint:
    uv run python -m hw1_imitation.playground --checkpoint path/to/checkpoint.pkl

    # Specify data directory for normalizer:
    uv run python -m hw1_imitation.playground --checkpoint path/to/checkpoint.pkl --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pygame
import torch

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import BasePolicy


ENV_ID = "gym_pusht/PushT-v0"
WINDOW_SIZE = 512  # The environment uses a 512x512 coordinate space


def load_checkpoint(checkpoint_path: str, device: torch.device) -> BasePolicy:
    """Load a trained policy from checkpoint."""
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def create_normalizer(data_dir: Path | None = None) -> Normalizer:
    """Create a normalizer from the dataset or use identity normalization."""
    if data_dir is not None:
        print(f"Loading normalizer stats from dataset in {data_dir}...")
        zarr_path = download_pusht(data_dir)
        states, actions, _ = load_pusht_zarr(zarr_path)
        normalizer = Normalizer.from_data(states, actions)
        print("Normalizer created from dataset.")
        return normalizer
    
    # Return an identity normalizer (no normalization)
    print("Using identity normalizer (no data dir provided).")
    return Normalizer(
        state_mean=np.zeros(5),
        state_std=np.ones(5),
        action_mean=np.zeros(2),
        action_std=np.ones(2),
    )


class InteractivePlayground:
    """Interactive playground for experimenting with Push-T policies."""

    def __init__(
        self,
        model: BasePolicy | None,
        normalizer: Normalizer,
        device: torch.device,
        chunk_size: int = 16,
        flow_num_steps: int = 10,
    ):
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.chunk_size = chunk_size
        self.flow_num_steps = flow_num_steps

        # Create environment with human rendering
        self.env = gym.make(
            ENV_ID,
            obs_type="state",
            render_mode="human",
            visualization_width=680,
            visualization_height=680,
        )
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        # State variables
        self.obs = None
        self.done = True
        self.paused = False
        self.manual_mode = model is None
        self.unlimited_time = True  # Ignore truncation (time limit)
        
        # Action chunking state
        self.action_chunk: np.ndarray | None = None
        self.chunk_index = chunk_size

        # Mouse state
        self.mouse_pressed = False
        self.last_mouse_pos = None
        
        # User pusher (second circle that you control with mouse)
        self.user_pusher = None
        self.user_pusher_shape = None
        self.user_pusher_target = None
        self.user_pusher_enabled = False  # Disabled by default
        if self.user_pusher_enabled:
            self._setup_user_pusher()

        # Stats
        self.episode_reward = 0.0
        self.max_reward = 0.0
        self.step_count = 0
        
        # Trajectory visualization
        self.show_trajectory = True  # Toggle for showing current action chunk
        self.current_trajectory: np.ndarray | None = None  # Current action chunk trajectory
        
        # Sampling mode (hold S to enter)
        self.sampling_mode = False
        self.num_sample_trajectories = 10  # Number of trajectories to sample
        self.sampled_trajectories: list[np.ndarray] = []  # Trajectories shown in sampling mode

    def reset(self):
        """Reset the environment."""
        self.obs, _ = self.env.reset()
        self.done = False
        self.action_chunk = None
        self.chunk_index = self.chunk_size
        self.episode_reward = 0.0
        self.max_reward = 0.0
        self.step_count = 0
        self.user_pusher_target = None
        
        # Clear trajectory visualizations
        self.sampled_trajectories.clear()
        self.current_trajectory = None
        self.sampling_mode = False
        
        # Re-create user pusher since physics space is reset (only if enabled)
        self.user_pusher = None
        self.user_pusher_shape = None
        if self.user_pusher_enabled:
            self._setup_user_pusher()
        
        print("\n--- Episode Reset ---")
        print(f"Mode: {'Manual' if self.manual_mode else 'Policy'}")
        print(f"Unlimited time: {'ON' if self.unlimited_time else 'OFF'}")

    def sample_action_chunk(self, obs: np.ndarray | None = None) -> np.ndarray | None:
        """Sample an action chunk from the policy without updating state."""
        if self.model is None:
            return None
        
        obs = obs if obs is not None else self.obs
        if obs is None:
            return None
            
        state = (
            torch.from_numpy(self.normalizer.normalize_state(obs))
            .float()
            .to(self.device)
        )
        with torch.no_grad():
            pred_chunk = (
                self.model.sample_actions(
                    state.unsqueeze(0), num_steps=self.flow_num_steps
                )
                .cpu()
                .numpy()[0]
            )
        action_chunk = self.normalizer.denormalize_action(pred_chunk)
        action_chunk = np.clip(action_chunk, self.action_low, self.action_high)
        return action_chunk

    def _sample_trajectories(self):
        """Sample multiple trajectories for visualization."""
        self.sampled_trajectories.clear()
        if self.model is None or self.obs is None:
            return
        
        # Get agent position to prepend to each trajectory
        agent_pos = np.array([[self.obs[0], self.obs[1]]])
        
        for _ in range(self.num_sample_trajectories):
            traj = self.sample_action_chunk()
            if traj is not None:
                # Prepend agent position so trajectory starts from agent
                traj_with_start = np.vstack([agent_pos, traj])
                self.sampled_trajectories.append(traj_with_start)

    def set_agent_position(self, pos: tuple[float, float]):
        """Set the agent position and update observation."""
        try:
            base_env = self.env.unwrapped
            # Set position and zero velocity
            base_env.agent.position = pos
            base_env.agent.velocity = (0, 0)
            # Step physics briefly to apply the change
            base_env.space.step(base_env.dt)
            # Get fresh observation from environment
            self.obs = base_env.get_obs()
        except Exception as e:
            print(f"Could not set agent position: {e}")

    def get_policy_action(self) -> np.ndarray:
        """Get action from the policy."""
        if self.model is None:
            # If no model, return current agent position (no movement)
            return np.array([self.obs[0], self.obs[1]])

        # Check if we need a new action chunk
        if self.action_chunk is None or self.chunk_index >= self.chunk_size:
            self.action_chunk = self.sample_action_chunk()
            self.chunk_index = 0
            # Update current trajectory for visualization (starts from agent position)
            if self.show_trajectory:
                agent_pos = np.array([[self.obs[0], self.obs[1]]])
                self.current_trajectory = np.vstack([agent_pos, self.action_chunk])

        action = self.action_chunk[self.chunk_index]
        self.chunk_index += 1
        return action

    def _setup_user_pusher(self):
        """Add a user-controlled pusher circle (same as agent) to the physics space."""
        if self.user_pusher is not None:
            return  # Already exists
        try:
            import pymunk
            base_env = self.env.unwrapped
            
            # Create a kinematic body (same as the agent) - a circle that can push the T
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            body.position = (100, 100)  # Start off to the side
            shape = pymunk.Circle(body, 15)  # Same radius as agent
            shape.color = (255, 100, 100, 255)  # Red color (RGBA)
            shape.friction = 1
            
            # Add to the physics space
            base_env.space.add(body, shape)
            self.user_pusher = body
            self.user_pusher_shape = shape
        except Exception as e:
            print(f"Could not create user pusher: {e}")
            self.user_pusher = None

    def _remove_user_pusher(self):
        """Remove the user pusher from the physics space."""
        try:
            if self.user_pusher is not None and self.user_pusher_shape is not None:
                base_env = self.env.unwrapped
                base_env.space.remove(self.user_pusher, self.user_pusher_shape)
                self.user_pusher = None
                self.user_pusher_shape = None
        except Exception as e:
            print(f"Could not remove user pusher: {e}")

    def mouse_to_env_coords(self, mouse_pos: tuple[int, int]) -> tuple[float, float]:
        """Convert mouse position to environment coordinates."""
        # The window is 512x512 in human render mode
        # (not 680x680 - that's only for rgb_array mode)
        env_x = float(mouse_pos[0])
        env_y = float(mouse_pos[1])
        # Clamp to valid range
        env_x = np.clip(env_x, 5, 507)
        env_y = np.clip(env_y, 5, 507)
        return env_x, env_y

    def update_user_pusher(self):
        """Move the user pusher to follow the mouse, or stay still."""
        if self.user_pusher is None:
            return
            
        try:
            if self.user_pusher_target is not None:
                # When mouse is pressed: move directly to mouse position
                self.user_pusher.position = self.user_pusher_target
            
            # Always keep velocity at zero (no momentum)
            self.user_pusher.velocity = (0, 0)
            
        except Exception as e:
            print(f"User pusher update error: {e}")

    def get_gradient_color(self, t: float) -> tuple:
        """Get color for position t (0=start, 1=end) in trajectory.
        
        Gradient: Blue -> Cyan -> Green -> Yellow -> Orange -> Red
        """
        # Define color stops: (position, (R, G, B))
        color_stops = [
            (0.0, (0, 0, 255)),      # Blue
            (0.2, (0, 200, 255)),    # Light blue / Cyan
            (0.4, (0, 255, 100)),    # Green
            (0.6, (255, 255, 0)),    # Yellow
            (0.8, (255, 165, 0)),    # Orange
            (1.0, (255, 0, 0)),      # Red
        ]
        
        # Find the two color stops to interpolate between
        for i in range(len(color_stops) - 1):
            t0, c0 = color_stops[i]
            t1, c1 = color_stops[i + 1]
            if t0 <= t <= t1:
                # Interpolate
                local_t = (t - t0) / (t1 - t0)
                r = int(c0[0] + (c1[0] - c0[0]) * local_t)
                g = int(c0[1] + (c1[1] - c0[1]) * local_t)
                b = int(c0[2] + (c1[2] - c0[2]) * local_t)
                return (r, g, b)
        
        return color_stops[-1][1]  # Default to red

    def draw_trajectory(self, surface: pygame.Surface, trajectory: np.ndarray):
        """Draw a trajectory with gradient colors (blue->cyan->green->yellow->orange->red)."""
        if trajectory is None or len(trajectory) == 0:
            return
        
        points = [(int(p[0]), int(p[1])) for p in trajectory]
        n = len(points)
        
        # Draw lines connecting the points with gradient colors
        for i in range(n - 1):
            t = i / max(1, n - 1)
            color = self.get_gradient_color(t)
            pygame.draw.line(surface, color, points[i], points[i + 1], 2)
        
        # Draw circles at each waypoint with gradient colors
        for i, point in enumerate(points):
            t = i / max(1, n - 1)
            color = self.get_gradient_color(t)
            # Small circles (radius 2-4)
            radius = max(2, 4 - (i * 2) // n)
            pygame.draw.circle(surface, color, point, radius)

    def draw_all_trajectories(self):
        """Draw all trajectories on the pygame window."""
        try:
            base_env = self.env.unwrapped
            if base_env.window is None:
                return
            
            surface = base_env.window
            
            # Draw sampled trajectories (in sampling mode)
            for traj in self.sampled_trajectories:
                self.draw_trajectory(surface, traj)
            
            # Draw current trajectory during runtime (not in sampling mode)
            if self.show_trajectory and self.current_trajectory is not None and not self.paused and not self.sampling_mode:
                self.draw_trajectory(surface, self.current_trajectory)
            
            pygame.display.update()
            
        except Exception as e:
            pass  # Silently ignore drawing errors

    def step(self):
        """Execute one environment step."""
        if self.done or self.paused or self.sampling_mode:
            return

        # Update user pusher target based on mouse (only when not in sampling mode)
        if self.mouse_pressed and self.last_mouse_pos and not self.sampling_mode:
            self.user_pusher_target = self.mouse_to_env_coords(self.last_mouse_pos)
        else:
            self.user_pusher_target = None
            
        # Move user pusher with same physics as agent
        self.update_user_pusher()

        # Get action from policy (or stay still in manual mode)
        if self.manual_mode:
            # In manual mode, agent stays in place
            action = np.array([self.obs[0], self.obs[1]], dtype=np.float32)
        else:
            action = self.get_policy_action()

        # Execute action
        self.obs, reward, terminated, truncated, info = self.env.step(
            action.astype(np.float32)
        )
        self.episode_reward += reward
        self.max_reward = max(self.max_reward, reward)
        self.step_count += 1
        
        # Ignore truncation if unlimited time is enabled
        if self.unlimited_time:
            truncated = False
        
        self.done = terminated or truncated

        if self.done:
            print(f"\n--- Episode Complete ---")
            print(f"Steps: {self.step_count}")
            print(f"Max Reward: {self.max_reward:.4f}")
            print(f"Total Reward: {self.episode_reward:.4f}")
            if terminated:
                print("Success! T-block reached goal.")
            else:
                print("Episode truncated (time limit).")

    def handle_event(self, event: pygame.event.Event):
        """Handle pygame events."""
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            elif event.key == pygame.K_SPACE:
                was_paused = self.paused
                self.paused = not self.paused
                if was_paused and not self.paused:
                    # Resuming - clear paused trajectories
                    self.paused_trajectories.clear()
                    print("Resumed - cleared trajectory samples")
                else:
                    print(f"Paused - press S to sample trajectories, SPACE to resume")
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_m:
                if self.model is not None:
                    self.manual_mode = not self.manual_mode
                    print(f"Mode: {'Manual' if self.manual_mode else 'Policy'}")
                else:
                    print("No model loaded - staying in manual mode")
            elif event.key == pygame.K_t:
                self.unlimited_time = not self.unlimited_time
                print(f"Unlimited time: {'ON' if self.unlimited_time else 'OFF'}")
            elif event.key == pygame.K_p:
                self.user_pusher_enabled = not self.user_pusher_enabled
                if self.user_pusher_enabled:
                    self._setup_user_pusher()
                    print("Red pusher: ON")
                else:
                    self._remove_user_pusher()
                    print("Red pusher: OFF")
            elif event.key == pygame.K_v:
                self.show_trajectory = not self.show_trajectory
                print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
            elif event.key == pygame.K_s:
                if self.model is not None:
                    self.sampling_mode = not self.sampling_mode
                    if self.sampling_mode:
                        self._sample_trajectories()
                        print(f"Sampling mode ON - drag agent to see {self.num_sample_trajectories} trajectories")
                    else:
                        self.sampled_trajectories.clear()
                        # Invalidate old action chunk so policy samples fresh from new position
                        self.action_chunk = None
                        self.chunk_index = self.chunk_size
                        print("Sampling mode OFF")
                else:
                    print("No model loaded")
            elif event.key == pygame.K_c:
                # Clear sampled trajectories
                self.sampled_trajectories.clear()
                print("Cleared trajectory samples")
        

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_pressed = True
                self.last_mouse_pos = pygame.mouse.get_pos()
                if self.sampling_mode:
                    # In sampling mode, drag the agent
                    pos = self.mouse_to_env_coords(self.last_mouse_pos)
                    self.set_agent_position(pos)
                    self._sample_trajectories()

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_pressed = False

        if event.type == pygame.MOUSEMOTION:
            self.last_mouse_pos = pygame.mouse.get_pos()
            if self.mouse_pressed:
                if self.sampling_mode:
                    # In sampling mode, drag the agent and resample
                    pos = self.mouse_to_env_coords(self.last_mouse_pos)
                    self.set_agent_position(pos)
                    self._sample_trajectories()

        return True

    def run(self):
        """Main loop."""
        pygame.init()
        clock = pygame.time.Clock()
        
        print("\n" + "="*50)
        print("Push-T Interactive Playground")
        print("="*50)
        print("\nControls:")
        print("  Left click + drag: Control red pusher OR drag agent (in sampling mode)")
        print("  S: Toggle sampling mode - drag agent to see trajectories")
        print("  P: Toggle red pusher on/off (default: OFF)")
        print("  C: Clear sampled trajectories")
        print("  V: Toggle trajectory visualization")
        print("  SPACE: Pause/resume")
        print("  R: Reset episode")
        print("  M: Toggle manual mode (agent stops)")
        print("  T: Toggle unlimited time (default: ON)")
        print("  Q/ESC: Quit")
        print("\nMulti-modality visualization:")
        print("  Press S to enter sampling mode, then drag the blue agent")
        print(f"  You'll see {self.num_sample_trajectories} sampled trajectories update in real-time!")
        print("="*50)

        self.reset()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            # Step environment
            self.step()
            
            # Render is handled by gym's human render mode
            self.env.render()
            
            # Draw trajectories on top
            self.draw_all_trajectories()

            # Auto-reset on episode end (with delay)
            if self.done:
                pygame.time.wait(1000)
                self.reset()

            clock.tick(30)  # 30 FPS

        self.env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive playground for Push-T policies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained policy checkpoint (.pkl file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory for creating normalizer (default: data)",
    )
    parser.add_argument(
        "--no-normalizer",
        action="store_true",
        help="Skip normalizer creation (use identity normalization)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Action chunk size (default: 8, should match training)",
    )
    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=10,
        help="Number of flow matching steps (for flow policies)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model if provided
    model = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_checkpoint(args.checkpoint, device)
        print(f"Model loaded: {type(model).__name__}")
    else:
        print("No checkpoint provided - running in manual mode only")

    # Create normalizer
    if args.no_normalizer or model is None:
        normalizer = create_normalizer(None)
    else:
        normalizer = create_normalizer(Path(args.data_dir))

    # Create and run playground
    playground = InteractivePlayground(
        model=model,
        normalizer=normalizer,
        device=device,
        chunk_size=args.chunk_size,
        flow_num_steps=args.flow_num_steps,
    )
    playground.run()


if __name__ == "__main__":
    main()
