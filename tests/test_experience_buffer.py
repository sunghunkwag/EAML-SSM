"""
Unit tests for the new PyTorch-based ExperienceBuffer.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experience.experience_buffer import ExperienceBuffer


@pytest.fixture
def buffer():
    """Create a default buffer for testing."""
    return ExperienceBuffer(max_size=10, device="cpu")


def test_buffer_initialization(buffer):
    """Test buffer initialization."""
    assert len(buffer) == 0
    assert buffer.max_size == 10
    assert buffer.device == torch.device("cpu")


def test_add_to_buffer(buffer):
    """Test adding single and multiple experiences."""
    obs1 = torch.randn(4)
    tgt1 = torch.randn(4)
    buffer.add(obs1.unsqueeze(0), tgt1.unsqueeze(0))
    assert len(buffer) == 1

    obs_batch = torch.randn(3, 4)
    tgt_batch = torch.randn(3, 4)
    buffer.add(obs_batch, tgt_batch)
    assert len(buffer) == 4


def test_buffer_max_size(buffer):
    """Test that the buffer respects the max_size constraint."""
    obs_batch = torch.randn(12, 4)  # More than max_size=10
    tgt_batch = torch.randn(12, 4)
    buffer.add(obs_batch, tgt_batch)

    assert len(buffer) == 10  # Should not exceed max_size

    # Check if the first two elements were dropped (FIFO)
    first_added_obs = obs_batch[0]
    buffer_obs_list = [item[0] for item in buffer.buffer]
    assert not any(torch.equal(first_added_obs, obs) for obs in buffer_obs_list)


def test_get_batch_empty(buffer):
    """Test sampling from an empty buffer."""
    batch = buffer.get_batch(5)
    assert batch is None


def test_get_batch_sampling(buffer):
    """Test sampling from a populated buffer."""
    obs_batch = torch.randn(8, 4)
    tgt_batch = torch.randn(8, 4)
    buffer.add(obs_batch, tgt_batch)

    # Test sampling less than size
    obs_sample, tgt_sample = buffer.get_batch(5)
    assert obs_sample.shape == (5, 4)
    assert tgt_sample.shape == (5, 4)

    # Test sampling more than size
    obs_sample, tgt_sample = buffer.get_batch(20)
    assert obs_sample.shape == (8, 4)  # Should cap at buffer length
    assert tgt_sample.shape == (8, 4)


def test_buffer_device(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Test that tensors are stored on the correct device."""
    if device == "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for device test")

    cuda_buffer = ExperienceBuffer(max_size=10, device=device)
    obs_cpu = torch.randn(2, 4)
    tgt_cpu = torch.randn(2, 4)

    cuda_buffer.add(obs_cpu, tgt_cpu)

    # Check item in buffer
    item_obs, item_tgt = cuda_buffer.buffer[0]
    assert item_obs.device.type == device
    assert item_tgt.device.type == device

    # Check sampled batch
    obs_batch, tgt_batch = cuda_buffer.get_batch(2)
    assert obs_batch.device.type == device
    assert tgt_batch.device.type == device
