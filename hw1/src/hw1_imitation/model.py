"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from email import policy
from typing import Literal, TypeAlias

from numpy import concatenate
import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        linear_layer_list = []
        dim_list = [self.state_dim, *hidden_dims, self.action_dim * self.chunk_size]
        for i in range(len(dim_list) - 1):
            linear_layer_list.append(nn.Linear(dim_list[i], dim_list[i+1]))
            if i == len(dim_list) - 2:
                continue
            relu = nn.ReLU()
            linear_layer_list.append(relu)
        self.mlp = nn.Sequential(*linear_layer_list)

        self.loss = torch.nn.MSELoss(reduction='none')

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        policy_actions = self.sample_actions(state)
        mse_error = self.loss(policy_actions, action_chunk)
        return mse_error.sum()


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        policy_actions = self.mlp(state)
        return policy_actions.view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        linear_layer_list = []
        dim_list = [self.state_dim + self.action_dim * self.chunk_size + 1,
                     *hidden_dims,
                     self.action_dim * self.chunk_size]
        for i in range(len(dim_list) - 1):
            linear_layer_list.append(nn.Linear(dim_list[i], dim_list[i+1]))
            if i == len(dim_list) - 2:
                continue
            relu = nn.ReLU()
            linear_layer_list.append(relu)
        self.mlp = nn.Sequential(*linear_layer_list)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        time_step = torch.rand(B, 1)
        x_0 = torch.randn(B, self.chunk_size, self.action_dim)
        t = time_step.unsqueeze(-1)
        x_t = t * action_chunk + (1 - t) * x_0
        x_t_flat = x_t.view(-1, self.chunk_size * self.action_dim)
        v_target = action_chunk - x_0
        network_input = torch.cat([state, x_t_flat, time_step], dim=-1)
        v_pred = self.mlp(network_input).view(-1, self.chunk_size, self.action_dim)
        mse_error = torch.nn.functional.mse_loss(v_pred, v_target, reduction='sum')
        return mse_error

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        action = torch.randn(B, self.chunk_size, self.action_dim)
        for i in range(num_steps):
            t = torch.full((B, 1), i / num_steps)
            action_flat = action.view(-1, self.chunk_size * self.action_dim)
            network_input = torch.cat([state, action_flat, t], dim=-1)
            v_pred = self.mlp(network_input).view(-1, self.chunk_size, self.action_dim)
            action = action + (v_pred / num_steps)
        return action


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
