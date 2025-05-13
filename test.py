import os, re, subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from TestPilot.common_helpers import setup_logger
from TestPilot.common_helpers import cls_Settings
from TestPilot.test_result import cls_TestResult
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases

def _get_python_files(directory: str) -> Tuple[List[Path], int, List[dict]]:
    """
    Recursively collects all Python (.py) files from the given directory and
    returns them along with initialized counters for tracking test results.

    Args:
        directory (str): The root directory to search for Python files.

    Returns:
        Tuple[List[Path], int, List[Any]]: A tuple containing:
            - A list of .py files found
            - An integer initialized to 0 for passed test count
            - An empty list for storing test statistics
    """
    passed_count = 0
    test_stats = []
    return list(Path(directory).rglob("*.py")), passed_count, test_stats

from dotenv import load_dotenv
def main() -> None:
        load_dotenv(override=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(".").resolve())

        cls_settings=cls_Settings()
        source_code_dir = list(Path(cls_settings.finalized_tests_dir).rglob("*.py"))
        for test_file in source_code_dir:
            result = subprocess.run(
                ["pytest", str(test_file), "--tb=short", "--quiet"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            passed = result.returncode == 0
            err_msg = result.stdout.strip() if not passed else ""
            print(f"{test_file} - {passed}")
            if not passed:
                print(f"Err_msg - \n{err_msg}")



if __name__ == "__main__":
    main()
