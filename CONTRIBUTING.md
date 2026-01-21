# Contributing to observ-python

Thanks for your interest in contributing to the Observ Python SDK! This guide will help you understand our development process and how to contribute effectively.

## Automated Release Process

This repository uses **python-semantic-release** to automatically publish new versions when pull requests are merged to the main branch. This means:

- ✅ **No manual version bumps** - versions are determined automatically
- ✅ **Automatic changelog generation** - based on your commit messages  
- ✅ **Automatic PyPI publishing** - happens when PR is merged to main
- ✅ **GitHub releases** - created automatically with release notes

## Commit Message Format

We use **Conventional Commits** to determine version bumps and generate changelogs. Your commit messages must follow this format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat` | New feature | **Minor** (1.2.0 → 1.3.0) |
| `fix` | Bug fix | **Patch** (1.2.0 → 1.2.1) |
| `feat!` or `fix!` | Breaking change | **Major** (1.2.0 → 2.0.0) |
| `docs` | Documentation changes | None |
| `style` | Code style changes | None |
| `refactor` | Code refactoring | None |
| `test` | Test changes | None |
| `chore` | Maintenance tasks | None |

### Examples

```bash
# This will trigger a patch release (1.2.0 → 1.2.1)
git commit -m "fix: resolve JWT token parsing issue"

# This will trigger a minor release (1.2.0 → 1.3.0)
git commit -m "feat: add support for Mistral AI provider"

# This will trigger a major release (1.2.0 → 2.0.0)
git commit -m "feat!: remove deprecated with_options method"

# This will NOT trigger a release
git commit -m "docs: update README with new examples"
```

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/observ-ai/observ-python.git
cd observ-python

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Check linting
uv run ruff check .

# Check types
uv run mypy observ --ignore-missing-imports
```

### 2. Making Changes

1. **Create a feature branch** from `main`
2. **Make your changes** following Python best practices
3. **Write or update tests** for your changes
4. **Ensure all checks pass**:
   ```bash
   uv run pytest tests/ -v        # All tests pass
   uv run ruff check .            # No linting errors
   uv run mypy observ             # Type check passes
   uv build                       # Package builds successfully
   ```

### 3. Creating a Pull Request

1. **Write a clear PR title** using conventional commit format
2. **Describe your changes** in the PR description
3. **Link any related issues**
4. **Ensure CI passes** - all automated checks must pass

### 4. Release Process

When your PR is merged to `main`:

1. **python-semantic-release runs automatically**
2. **Version is determined** from commit messages since last release
3. **Package is built and published to PyPI**
4. **GitHub release is created** with generated changelog
5. **CHANGELOG.md is updated** and committed

## Code Standards

### Python Guidelines

- Follow **PEP 8** style guidelines (enforced by Ruff)
- Use **type hints** for all public APIs
- Write **clear docstrings** for modules, classes, and functions
- Follow **existing code patterns** and conventions
- Use **meaningful variable and function names**

### Code Quality Tools

- **Ruff**: Linting and code formatting
- **mypy**: Static type checking  
- **pytest**: Testing framework
- **uv**: Dependency management and virtual environments

### Testing

- Write **unit tests** for new features using pytest
- Ensure **good test coverage** of core functionality
- Mock external dependencies in tests
- Test both success and error scenarios
- Place tests in `tests/` directory

### Provider Integration

When adding new AI provider support:

1. **Create wrapper class** in `observ/providers/`
2. **Implement chainable methods** (`with_metadata`, `with_session_id`)
3. **Add provider method** to main `Observ` class
4. **Update `__init__.py`** to export the new wrapper
5. **Add to optional dependencies** in pyproject.toml
6. **Write tests** for the new provider

## Virtual Environment

This project uses **uv** for dependency management. Key commands:

```bash
# Install dependencies
uv sync

# Install with optional extras
uv sync --all-extras

# Add a dependency
uv add package_name

# Add a dev dependency  
uv add --dev package_name

# Run commands in the virtual environment
uv run pytest
uv run python script.py
```

## Testing Locally

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_basic.py -v

# Run with coverage
uv run pytest tests/ --cov=observ --cov-report=html

# Test installation locally
uv build
pip install dist/*.whl --force-reinstall
```

## Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/observ-ai/observ-python/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/observ-ai/observ-python/discussions)  
- **Documentation**: Check our [main documentation](https://docs.observ.dev)

## Release Notes

All releases are automatically documented in:
- **GitHub Releases**: https://github.com/observ-ai/observ-python/releases
- **CHANGELOG.md**: Generated automatically from commit messages
- **PyPI**: https://pypi.org/project/observ-sdk/

Thank you for contributing to Observ! 🚀