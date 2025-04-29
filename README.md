# ✅ Auto Unit Test Generator with Pre-commit Integration

This project automates the generation of pytest-style unit tests for Python files using OpenAI's large language models (LLMs). It integrates with pre-commit to automatically generate or update tests whenever a file in the `src/` directory is changed.

---

# 🎯 Objectives

- Automatically generate unit tests for updated Python files.
- Use OpenAI's GPT model for intelligent and comprehensive test generation.
- Save tests to the `tests/` directory in a mirrored folder structure.
- Stage test files for commit automatically.
- Run all unit tests after generation using `pytest`.

---

# ⚙️ Key Features

## 🧠 LLM-Powered Test Generation
- Uses OpenAI's `gpt-4-turbo` or configurable model from `.env`.
- Prompts include extracted imports and function names for context.
- Ensures all public functions are tested with meaningful test names.

## 🧪 Pre-commit Hook Integration
- Automatically triggers test generation when files in `src/` change.
- Stages newly created test files using `git add`.
- Runs all tests via `pytest` after generation.
- Includes CLI logging to track processing and errors.

## 📁 Test File Management
- Tests saved in `tests/` directory with mirrored structure:
  e.g., `src/module.py` → `tests/test_module.py`.
- Ensures consistent naming conventions and UTF-8 encoding.

## 🌐 Environment Variables via `.env`
- Configure source and test directories.
- Model name and LLM prompt template are customizable.
- Example:
  ```env
  OPENAI_API_KEY=your_openai_key
  SRC_DIR=./src
  TESTS_DIR=./tests
  MODEL_NAME=gpt-4-turbo
  LLM_TEST_PROMPT_TEMPLATE=...


  0. ✅ Only write tests if the file contains executable logic, such as:
Functions or class definitions

Code blocks with behavior

❌ Do NOT generate tests for files that only contain import statements (e.g., __init__.py)

1. ✅ At the top of the test file, include all required imports:
{import_section} for original file imports

{import_hint} to import the functions or classes under test

Always include import pytest if it is used in the test code

2. ✅ Write tests for all public functions and classes in the source file
3. ✅ Use pytest-style assertions throughout
For float comparisons, use: pytest.approx(expected)

4. ✅ Use unittest.mock (or pytest-mock) for mocking external dependencies such as:
File I/O

API calls

Database access

Network requests

5. ✅ Write clear and descriptive test function names using snake_case
6. ✅ For return values from mocks, prefer behavior checks like:
hasattr()

Method calls

❌ Avoid isinstance() for these checks

7. ❌ Do NOT include any of the following:
Explanations

Docstrings

Comments

Markdown formatting (no triple backticks)

📄 Input file path: {file_path}
📦 Python source code to test:
{file_content}
🧪 Output:
Generate a single test file that is complete and executable with pytest.

$ uv init
$ uv venv
$ source .venv\\Scripts\\activate
$ uv add pre-commit
$ pre-commit install
$ uv pip install -r requirements.txt


PYTHONPATH=. pytest ./temp/temp.py
sh ./scripts/quick-commit.sh

clear packages
python -m venv unit_test_env
source unit_test_env/Scripts/activate
pip list  # should be minimal: pip, setuptools, wheel