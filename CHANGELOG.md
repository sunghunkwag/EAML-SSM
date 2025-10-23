# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-23

### Added
- Initial release of EAML-SSM (Experience-Augmented Meta-Learning with State-Space Models)
- Integrated `ExperienceBuffer` component for storing and sampling past experiences
- Modified `Adapter` class to support hybrid-loss adaptation using both current task data and past experiences
- Complete test suite with 23 unit tests covering all major components
- CI/CD workflow for automated testing on multiple Python versions
- Comprehensive documentation in README.md

### Components
- **core/ssm.py**: PyTorch-based State-Space Model
- **meta_rl/meta_maml.py**: MAML implementation for stateful models
- **experience/experience_buffer.py**: PyTorch-based circular buffer for experience storage
- **adaptation/test_time_adaptation.py**: Hybrid test-time adaptation module
- **env_runner/environment.py**: Environment wrapper for RL tasks
- **main.py**: Main orchestration script for training and adaptation

### Features
- Meta-training with MAML for quick task adaptation
- Experience buffer population during data collection
- Hybrid adaptation combining current task loss and experience-based loss
- Configurable experience weight and batch size
- Gradient clipping for stable training
- Support for both discrete and continuous action spaces

### Testing
- All tests passing (22 passed, 1 skipped)
- Code formatted with Black
- Compatible with Python 3.8, 3.9, and 3.10

