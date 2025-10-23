# Contributing to EAML-SSM

Thank you for your interest in contributing to EAML-SSM! This document provides guidelines and instructions for contributing to the project.

## Development Setup

To set up your development environment, follow these steps:

```bash
git clone https://github.com/sunghunkwag/EAML-SSM.git
cd EAML-SSM
pip install -e .[dev]
```

This will install all necessary dependencies including development tools like pytest, black, and flake8.

## Code Style

This project follows the PEP 8 style guide with some modifications. We use Black for automatic code formatting with a maximum line length of 120 characters. Before submitting your code, please run:

```bash
black --line-length 120 .
```

## Testing

All new features and bug fixes should include appropriate tests. We use pytest for testing. To run the test suite:

```bash
pytest -v
```

Ensure that all tests pass before submitting a pull request. The test coverage should be maintained or improved with your contributions.

## Pull Request Process

When submitting a pull request, please follow these guidelines:

1. **Fork the repository** and create your branch from `main`
2. **Write clear commit messages** that describe the changes you made
3. **Update documentation** if you are adding new features or changing existing functionality
4. **Add tests** for any new code you introduce
5. **Run the test suite** to ensure all tests pass
6. **Format your code** using Black before submitting

Your pull request should include a clear description of the problem you are solving and the approach you took. If your PR addresses an existing issue, please reference it in the description.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. When reporting bugs, include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment details (Python version, OS, etc.)

## Code of Conduct

Please be respectful and constructive in all interactions with the community. We are committed to providing a welcoming and inclusive environment for all contributors.

