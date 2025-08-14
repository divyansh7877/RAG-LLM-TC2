# Testing Guide

## Quick Start

### 1. Setup Environment

```bash
# Install test dependencies
pip install -r test-requirements.txt

# Ensure you're in the project root directory
cd /path/to/your/project
```

### 2. Run Tests

#### Option A: Using the Test Runner (Recommended)

```bash
# Run all tests
python run_tests.py

# Run specific test suites
python run_tests.py --suite unit
python run_tests.py --suite integration  
python run_tests.py --suite performance
python run_tests.py --suite security

# Run with coverage report
python run_tests.py --coverage

# Run with verbose output
python run_tests.py --verbose

# Run tests in parallel (faster)
python run_tests.py --parallel
```

#### Option B: Using Pytest Directly

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_security_isolation.py -v
python -m pytest tests/test_performance_optimization.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### 3. Test Categories

#### Unit Tests
- `test_security_isolation.py` - User isolation and security
- `test_performance_optimization.py` - Resource management
- `test_api_endpoints_comprehensive.py` - API functionality
- Plus existing unit tests for core components

#### Integration Tests
- `test_integration_workflows.py` - End-to-end workflows
- Multi-user concurrent scenarios
- Error recovery testing

#### Performance Tests
- `test_performance_benchmarks.py` - Load testing
- Concurrent user simulation (50+ users)
- Resource pressure testing
- Performance regression detection

#### Security Tests
- `test_security_penetration.py` - Vulnerability testing
- Authentication security
- Data isolation verification
- Attack simulation

## Test Results

### Expected Output
```
Running: Unit Tests (unit)
Command: python -m pytest tests/test_security_isolation.py tests/test_performance_optimization.py tests/test_api_endpoints_comprehensive.py -v
============================================================
tests/test_security_isolation.py::TestUserDataIsolation::test_concurrent_user_query_isolation PASSED
tests/test_performance_optimization.py::TestResourceManagement::test_concurrent_embedding_limit_enforcement PASSED
...
✅ All tests passed!
```

### Coverage Report
After running with `--coverage`, check `htmlcov/index.html` for detailed coverage report.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r test-requirements.txt
   ```

3. **Redis Connection Errors**
   - Tests use mocked Redis, so no actual Redis server needed
   - If you see Redis errors, ensure mocking is working properly

4. **Slow Tests**
   ```bash
   # Run only fast tests
   python -m pytest -m "not slow" tests/
   
   # Or run tests in parallel
   python run_tests.py --parallel
   ```

## Test Development

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_*.py` file
2. **Integration Tests**: Add to `test_integration_workflows.py`
3. **Performance Tests**: Add to `test_performance_benchmarks.py`
4. **Security Tests**: Add to `test_security_penetration.py`

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
@pytest.mark.security
@pytest.mark.slow
def test_my_feature():
    pass
```

### Mocking Guidelines

- Mock external dependencies (Redis, file system, network)
- Use `unittest.mock.patch` for dependency injection
- Mock at the boundary of your system under test

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    - name: Run tests
      run: python run_tests.py --coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Performance Benchmarks

The test suite includes performance benchmarks with these targets:

- **Authentication**: < 100ms average response time
- **Session Management**: > 100 operations/second
- **Job Processing**: < 10ms for job creation/updates
- **Concurrent Users**: Support 50+ simultaneous users
- **Memory Usage**: Stable under load testing

## Security Testing

Security tests verify:

- JWT token security and expiration
- User data isolation
- Session security
- Input validation
- File upload security
- Protection against common attacks (brute force, privilege escalation)

Run security tests regularly:
```bash
python run_tests.py --suite security
```

## Getting Help

1. Check test output for specific error messages
2. Review `TEST_SUITE_SUMMARY.md` for detailed test documentation
3. Run individual test files to isolate issues
4. Use `--verbose` flag for detailed test output