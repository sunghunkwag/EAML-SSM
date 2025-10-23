"""
Test-Time Adaptation Module for EAML-SSM

This version is MODIFIED to accept an ExperienceBuffer and perform
hybrid-loss adaptation using both current data and past experiences.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# Import the new ExperienceBuffer
from experience.experience_buffer import ExperienceBuffer


# AdaptationConfig (Unchanged)
@dataclass
class AdaptationConfig:
    """Configuration for the Adapter."""

    learning_rate: float = 0.01
    num_steps: int = 5
    grad_clip_norm: Optional[float] = 1.0

    # New config fields for hybrid adaptation
    experience_batch_size: int = 32
    experience_weight: float = 0.1  # Weight for the experience-based loss


class Adapter:
    """
    Performs test-time adaptation using a hybrid loss.

    This Adapter is enhanced to leverage an ExperienceBuffer, combining
    the loss from the current task with a loss from sampled past experiences.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptationConfig,
        experience_buffer: ExperienceBuffer,  # MODIFIED: Accept buffer
        device: str = "cpu",
    ):

        if torch is None:
            raise RuntimeError("PyTorch is required for test-time adaptation")

        self.model = model
        self.config = config
        self.device = device

        # --- MODIFIED: Store the experience buffer ---
        self.experience_buffer = experience_buffer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_step(
        self, x_current: torch.Tensor, y_current: torch.Tensor, hidden_state_current: torch.Tensor
    ) -> Tuple[float, int]:
        """
        Performs adaptation update steps using a hybrid loss.

        Args:
            x_current: Input tensor for the CURRENT task (B_curr, input_dim)
            y_current: Target tensor for the CURRENT task (B_curr, output_dim)
            hidden_state_current: Current hidden state for the CURRENT task (B_curr, state_dim)

        Returns:
            Tuple[float, int]:
                - loss (float): The total hybrid loss from the final step.
                - steps (int): The number of steps taken.
        """

        self.model.train()
        total_loss_item = 0.0

        for step in range(self.config.num_steps):

            self.optimizer.zero_grad()

            # --- 1. Loss on CURRENT task data ---
            # Forward pass for the current, stateful data
            output_current, next_hidden_state = self.model(x_current, hidden_state_current)
            loss_current = self.loss_fn(output_current, y_current)

            total_loss = loss_current

            # --- 2. Loss on PAST experience data ---
            experience_batch = self.experience_buffer.get_batch(self.config.experience_batch_size)

            if experience_batch is not None:
                x_exp, y_exp = experience_batch

                # We assume past experiences are independent sequences.
                # Thus, we initialize a new hidden state for them.
                B_exp = x_exp.shape[0]
                hidden_state_exp = self.model.init_hidden(batch_size=B_exp)

                # Forward pass for the stateless experience batch
                output_exp, _ = self.model(x_exp, hidden_state_exp)
                loss_experience = self.loss_fn(output_exp, y_exp)

                # --- 3. Combine losses ---
                total_loss = loss_current + self.config.experience_weight * loss_experience

            # 4. Backpropagation on combined loss
            total_loss.backward()

            # 5. Gradient Clipping (optional, from original config)
            if self.config.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            # 6. Optimizer step
            self.optimizer.step()

            # 7. Detach the *current* hidden state for the next loop iteration
            hidden_state_current = next_hidden_state.detach()

            total_loss_item = total_loss.item()

        # Return the final total loss and steps taken
        return total_loss_item, self.config.num_steps
