# CoreCoder Baseline

## 1. Development Environment

- Operating system: Windows
- IDE: Visual Studio Code
- Python virtual environment: agent_env
- Development branch: devpilot-v1
- LLM provider: DeepSeek OpenAI-compatible API
- Model: deepseek-v4-flash

## 2. Baseline Verification

The original CoreCoder project was installed and verified before DevPilot
development started.

Verification results:

- Editable installation: passed
- Pytest: 86 passed
- Ruff: passed
- Python compileall: passed
- Basic LLM response: passed
- read_file tool call: passed
- Automatic code repair experiment: passed

## 3. Automatic Repair Experiment

A temporary calculator example was created with an intentional bug:

```python
def add(a: int, b: int) -> int:
    return a - b