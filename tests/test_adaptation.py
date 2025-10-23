# -*- coding: utf-8 -*-
"""
Unit tests for the MODIFIED Test-Time Adaptation implementation.
Tests that the Adapter correctly uses the ExperienceBuffer.
"""
import pytest
import torch
import torch.nn as nn
import copy
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adaptation.test_time_adaptation import Adapter, AdaptationConfig
from core.ssm import StateSpaceModel
from experience.experience_buffer import ExperienceBuffer


class TestHybridAdapter:
    """Test suite for the new hybrid Adapter."""

    @pytest.fixture
    def model(self):
        """Create a simple test model."""
        # Input dim 4, Output dim 4 (obs -> next_obs)
        return StateSpaceModel(state_dim=4, input_dim=4, output_dim=4)

    @pytest.fixture
    def experience_buffer(self):
        """Create a mock, populated experience buffer."""
        buffer = ExperienceBuffer(max_size=100, device="cpu")
        obs_batch = torch.randn(50, 4)
        tgt_batch = torch.randn(50, 4)
        buffer.add(obs_batch, tgt_batch)
        return buffer

    @pytest.fixture
    def empty_buffer(self):
        """Create an empty buffer."""
        return ExperienceBuffer(max_size=100, device="cpu")

    @pytest.fixture
    def config(self):
        """Create AdaptationConfig."""
        return AdaptationConfig(learning_rate=0.01, num_steps=5, experience_batch_size=10, experience_weight=0.1)

    def test_adapter_initialization(self, model, config, experience_buffer):
        """Test that Adapter initializes correctly with the buffer."""
        adapter = Adapter(model=model, config=config, experience_buffer=experience_buffer)
        assert adapter.model is model
        assert adapter.config is config
        assert adapter.experience_buffer is experience_buffer
        print("✓ Adapter initialization with buffer successful")

    def _get_params_mean_std(self, model):
        """Helper to get model param stats."""
        params = [p.data.flatten() for p in model.parameters()]
        flat_params = torch.cat(params)
        return flat_params.mean().item(), flat_params.std().item()

    def test_parameter_mutation_with_buffer(self, model, config, experience_buffer):
        """
        Test that update_step() mutates parameters when using the buffer.
        """
        adapter = Adapter(model=model, config=config, experience_buffer=experience_buffer)

        x = torch.randn(1, 4)
        y = torch.randn(1, 4)
        hidden_state = model.init_hidden(batch_size=1)

        mean_before, std_before = self._get_params_mean_std(model)

        # Perform update step
        adapter.update_step(x_current=x, y_current=y, hidden_state_current=hidden_state)

        mean_after, std_after = self._get_params_mean_std(model)

        print(f"Params Before: mean={mean_before:.6f}, std={std_before:.6f}")
        print(f"Params After:  mean={mean_after:.6f}, std={std_after:.6f}")

        assert (
            mean_before != mean_after or std_before != std_after
        ), "Parameters did not change after update step with buffer."
        print("✓ Parameter mutation with buffer verified")

    def test_parameter_mutation_with_empty_buffer(self, model, config, empty_buffer):
        """
        Test that update_step() still mutates parameters (using current_loss only)
        when the buffer is empty.
        """
        adapter = Adapter(model=model, config=config, experience_buffer=empty_buffer)

        x = torch.randn(1, 4)
        y = torch.randn(1, 4)
        hidden_state = model.init_hidden(batch_size=1)

        mean_before, std_before = self._get_params_mean_std(model)

        # Perform update step
        adapter.update_step(x_current=x, y_current=y, hidden_state_current=hidden_state)

        mean_after, std_after = self._get_params_mean_std(model)

        print(f"Params Before (empty buffer): mean={mean_before:.6f}, std={std_before:.6f}")
        print(f"Params After (empty buffer):  mean={mean_after:.6f}, std={std_after:.6f}")

        assert (
            mean_before != mean_after or std_before != std_after
        ), "Parameters did not change after update step with empty buffer."
        print("✓ Parameter mutation with empty buffer verified")

    def test_loss_value_return(self, model, config, experience_buffer):
        """Test that a valid loss value is returned."""
        adapter = Adapter(model=model, config=config, experience_buffer=experience_buffer)
        x = torch.randn(1, 4)
        y = torch.randn(1, 4)
        hidden_state = model.init_hidden(batch_size=1)

        loss, steps = adapter.update_step(x_current=x, y_current=y, hidden_state_current=hidden_state)

        assert isinstance(loss, float)
        assert loss > 0.0
        assert steps == config.num_steps
        print(f"✓ Valid loss returned: {loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
