# Contributing to Two-Stage Latent ODE

Thank you for your interest in contributing to the Two-Stage Latent ODE project! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Two-stage-latent-ODE.git
   cd Two-stage-latent-ODE
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting:
  ```bash
  pytest tests/
  ```

### Documentation

- Update README.md if adding new features
- Add docstrings to all new functions/classes
- Include usage examples for new functionality

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Submit a Pull Request on GitHub

## Pull Request Guidelines

- Provide a clear description of the changes
- Include test coverage for new features
- Ensure all tests pass
- Update documentation as needed
- Follow the existing code style

## Areas for Contribution

We welcome contributions in the following areas:

### Model Enhancements
- Additional encoder architectures (Transformers, attention mechanisms)
- Alternative ODE solvers or integration methods
- Multi-modal inputs (glucose, insulin, meals, activity)

### Training Improvements
- Advanced loss functions
- Curriculum learning strategies
- Data augmentation techniques

### Evaluation
- Additional clinical metrics
- Benchmarking against baselines
- Visualization improvements

### Documentation
- Tutorial notebooks
- Usage examples
- Architecture explanations

### Testing
- Integration tests
- Performance benchmarks
- Edge case coverage

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about usage
- Discussion of potential improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
