"""Model definitions using Facebook Research flow_matching library.

This module provides FlowMatchingPolicy implementations using the
flow_matching library from https://github.com/facebookresearch/flow_matching

The original manual implementation is preserved in model.py.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

from hw1_imitation.model import BasePolicy, MSEPolicy


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> nn.Sequential:
    """Build an MLP with ReLU activations."""
    layers: list[nn.Module] = []
    dim_list = [input_dim, *hidden_dims, output_dim]
    for i in range(len(dim_list) - 1):
        layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
        if i < len(dim_list) - 2:  # No activation on final layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FlowMatchingPolicyFM(BasePolicy):
    """Flow matching policy using Facebook Research flow_matching library.
    
    This implementation uses:
    - CondOTProbPath for computing interpolated samples and target velocities
    - ODESolver for sampling actions during inference
    
    CondOTProbPath uses conditional optimal transport with:
        alpha_t = t, sigma_t = 1 - t
    
    This produces:
        x_t = t * x_1 + (1 - t) * x_0
        dx_t = x_1 - x_0
    
    This is mathematically equivalent to the manual implementation in model.py.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        # Velocity network: takes [state, x_t_flat, t] -> velocity
        input_dim = state_dim + action_dim * chunk_size + 1
        output_dim = action_dim * chunk_size
        self.mlp = build_mlp(input_dim, output_dim, hidden_dims)
        
        # Flow matching path from library (CondOT = conditional optimal transport)
        self.path = CondOTProbPath()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss using the library's path sampling.
        
        Args:
            state: Batch of states, shape (B, state_dim)
            action_chunk: Batch of action chunks, shape (B, chunk_size, action_dim)
            
        Returns:
            Scalar loss value (sum of MSE over all dimensions)
        """
        B = state.shape[0]
        device = state.device
        
        # Sample random time for each batch element (shape: (B,) for the library)
        t = torch.rand(B, device=device)
        
        # Sample noise (source distribution)
        x_0 = torch.randn_like(action_chunk)
        
        # Target actions (target distribution)
        x_1 = action_chunk
        
        # Use library's path sampling to get interpolated sample and target velocity
        # path.sample takes x_0, x_1, t and returns PathSample with x_t and dx_t
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t  # Interpolated sample at time t
        dx_t = path_sample.dx_t  # Target velocity (derivative of path)
        
        # Prepare network input: concatenate [state, x_t_flat, t]
        x_t_flat = x_t.view(B, -1)
        t_input = t.view(B, 1)
        network_input = torch.cat([state, x_t_flat, t_input], dim=-1)
        
        # Predict velocity
        v_pred = self.mlp(network_input).view(B, self.chunk_size, self.action_dim)
        
        # Compute MSE loss between predicted and target velocity
        loss = F.mse_loss(v_pred, dx_t, reduction="sum")
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample actions using ODESolver from the flow_matching library.
        
        Args:
            state: Batch of states, shape (B, state_dim)
            num_steps: Number of integration steps
            
        Returns:
            Sampled action chunks, shape (B, chunk_size, action_dim)
        """
        B = state.shape[0]
        device = state.device
        
        # Create velocity function that the solver will call
        # The solver expects velocity_fn(t, x) -> dx/dt
        def velocity_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Velocity function for ODE solver.
            
            Args:
                t: Current time, scalar tensor
                x: Current state, shape (B, chunk_size, action_dim)
                
            Returns:
                Predicted velocity, same shape as x
            """
            x_flat = x.view(B, -1)
            # Expand scalar t to batch dimension
            t_batch = t.expand(B).view(B, 1)
            network_input = torch.cat([state, x_flat, t_batch], dim=-1)
            v_pred = self.mlp(network_input)
            return v_pred.view(B, self.chunk_size, self.action_dim)
        
        # Create ODE solver with our velocity function
        solver = ODESolver(velocity_model=velocity_fn)
        
        # Sample from noise (source distribution)
        x_0 = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        
        # Integrate from t=0 to t=1 using the solver
        # step_size = 1/num_steps gives us the same discretization as manual Euler
        x_1 = solver.sample(x_init=x_0, step_size=1.0 / num_steps)
        
        return x_1


PolicyTypeFM: TypeAlias = Literal["mse", "flow_fm"]


def build_policy_fm(
    policy_type: PolicyTypeFM,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    """Build a policy using the flow_matching library implementation.
    
    Args:
        policy_type: Type of policy - "mse" for MSE baseline, "flow_fm" for flow matching
        state_dim: Dimension of state observations
        action_dim: Dimension of actions
        chunk_size: Number of actions per chunk
        hidden_dims: Hidden layer dimensions for the MLP
        
    Returns:
        Policy instance
    """
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow_fm":
        return FlowMatchingPolicyFM(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
