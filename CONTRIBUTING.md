# Contributing to Offline Wikipedia RAG

Thank you for your interest in contributing! This project aims to make AI with Wikipedia knowledge accessible to everyone.

## Ways to Contribute

### 1. Report Issues
- Bug reports
- Feature requests
- Documentation improvements
- Performance issues

### 2. Code Contributions
- New features
- Bug fixes
- Performance optimizations
- Test coverage

### 3. Documentation
- Improve README
- Add tutorials
- Write guides for different use cases
- Translate documentation

### 4. Language Support
- Add support for other Wikipedia languages (German, French, Spanish, etc.)
- Improve multilingual handling

## Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/offline-wikipedia-rag.git
cd offline-wikipedia-rag

# Create development environment
conda env create -f environment.yml
conda activate wikipedia-rag

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/amazing-feature`
3. **Make changes** and commit: `git commit -m 'Add amazing feature'`
4. **Push** to your fork: `git push origin feature/amazing-feature`
5. **Open a Pull Request** with a clear description

### PR Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow existing code style
- Keep PRs focused on a single feature/fix

## Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions small and focused

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_rag.py

# Run with coverage
python -m pytest --cov=.
```

## Areas for Contribution

### High Priority
- [ ] GPU acceleration support
- [ ] Docker container
- [ ] GUI interface (web or desktop)
- [ ] Additional language support

### Medium Priority
- [ ] Smaller model options (for lower RAM systems)
- [ ] Search result caching
- [ ] Conversation history
- [ ] Export answers to file

### Low Priority
- [ ] Custom prompts
- [ ] Answer formatting options
- [ ] Statistics tracking

## Questions?

Open an issue or start a discussion. We're happy to help!

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build together.
