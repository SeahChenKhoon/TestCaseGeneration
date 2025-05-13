import shutil
import pandas as pd

from typing import List, Optional, Tuple
from pathlib import Path
from tabulate import tabulate

from TestPilot.common_helpers import cls_Settings, SaveFile, LLMPromptExecutor, setup_logger
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases
from TestPilot.test_result import cls_TestResult

logger = setup_logger()

def _clean_test_environment(cls_settings: cls_Settings) -> None:
    """
    Cleans and resets the test environment by:
    1. Clearing the contents of temporary test files and logs.
    2. Recreating (i.e., deleting and re-initializing) the output directories 
       for generated, finalized, and failed tests.

    Args:
        cls_settings (cls_Settings): An object containing file paths and directory 
        configurations related to the test environment.
    """

    def _reset_file(file_path: str) -> None:
        """
        Creates the parent directory if needed and resets the file to empty.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def _recreate_dir(dir_path: str) -> None:
        """
        Removes the directory if it exists and recreates it empty.
        """
        path = Path(dir_path)
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    _reset_file(cls_settings.temp_test_file)
    _reset_file(cls_settings.log_file)

    for dir_path in [
        cls_settings.generated_tests_dir,
        cls_settings.finalized_tests_dir,
        cls_settings.failed_tests_dir,
    ]:
        _recreate_dir(dir_path)


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


def _run_initial_setup() -> Tuple[cls_Settings, List[str], cls_Test_Cases]:
    """
    Performs the initial setup for the test case generation process.

    This includes:
    - Initializing the settings object.
    - Cleaning up the test environment.
    - Parsing the source directory list from settings.
    - Initializing the test case holder object.

    Returns:
        Tuple[cls_Settings, List[str], cls_Test_Cases]: A tuple containing the
        settings object, a list of source directory paths, and the test case container.
    """
    cls_settings = cls_Settings()
    _clean_test_environment(cls_settings)
    source_dir_list = [path.strip() for path in cls_settings.source_dir_str.split(",") if path]
    cls_test_cases = cls_Test_Cases()
    
    return cls_settings, source_dir_list, cls_test_cases


def _process_and_save_test_results(
    source_dir: str,
    cls_source_code: cls_SourceCode,
    cls_settings: cls_Settings,
    successful_test_result: cls_TestResult,
    passed_count: int,
    overall_error_msg: Optional[str] = None
) -> str:
    """
    Finalizes and saves test results by:
    - Organizing imports via LLM.
    - Saving finalized test cases to disk.
    - Optionally logging any error messages.
    - Re-running the test to confirm final compilation status.

    Args:
        source_dir (str): The original source directory path.
        cls_source_code (cls_SourceCode): The source code object being tested.
        cls_settings (cls_Settings): Configuration settings including test folders and prompts.
        successful_test_result (cls_TestResult): Object containing the final test code.
        passed_count (int): Number of test cases that passed during generation.
        overall_error_msg (Optional[str], optional): Aggregated error message from retries.

    Returns:
        str: A message indicating whether the finalized test compiled and ran without errors.
    """
    savefile = SaveFile(source_dir, cls_source_code.source_code_file_path)
    is_passed: bool = False
    err_msg: str = ""

    if passed_count > 0:
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {"full_unit_tests": successful_test_result.full_test_case}

        successful_test_result.import_statement = llm_prompt_executor.execute_llm_prompt(
            cls_settings.llm_organize_imports_prompt, llm_parameter
        )

        savefile.save_file(
            Path(cls_settings.finalized_tests_dir),
            successful_test_result.full_test_case
        )

        is_passed, err_msg = successful_test_result.run_unit_test(
            successful_test_result.full_test_case, cls_settings, cls_source_code
        )

    if overall_error_msg:
        savefile.save_file(
            Path(cls_settings.failed_tests_dir),
            overall_error_msg,
            prefix="err_",
            file_extension=".log"
        )

    return "No Error in compilation" if is_passed else f"Error in compilation - {err_msg}"


def init_test_result(cls_test_cases: cls_Test_Cases) -> cls_TestResult:
    """
    Initializes a cls_TestResult instance based on the given test cases.

    Copies import statements and pytest fixtures from the input test case object,
    initializes a blank unit test list with a placeholder, and returns a
    cls_TestResult object with test_case_no set to 0.

    Args:
        cls_test_cases (cls_Test_Cases): Source object containing test structure.

    Returns:
        cls_TestResult: A new test result object with base test case info initialized.
    """
    test_case = cls_Test_Cases()
    test_case.import_statement = cls_test_cases.import_statement
    test_case.pytest_fixtures = cls_test_cases.pytest_fixtures
    test_case.unit_test = [""]  

    test_result = cls_TestResult(0, test_case)
    return test_result


def main() -> None:
    """
    Entry point for test case generation and evaluation.

    - Initializes configuration and environment.
    - Iterates through source directories to collect `.py` files.
    - Generates unit test cases using LLMs or predefined files.
    - Evaluates each generated test case, retries on failure.
    - Aggregates and logs test statistics in a tabular format.
    """
    cls_settings, source_dir_list, cls_test_cases = _run_initial_setup()

    for source_dir in source_dir_list:
        logger.info(f"Executing directory {source_dir}...")

        source_code_dir, passed_count, test_stats = _get_python_files(source_dir)

        for source_code_file in source_code_dir:
            cls_test_cases = cls_Test_Cases()

            passed_count, overall_error_msg, successful_import_stmt, \
            success_unit_test, cls_source_code = cls_test_cases.init_variables(
                source_code_file, cls_settings
            )

            total_test_case = cls_test_cases.derive_test_cases(
                source_dir, cls_source_code, cls_settings
            )

            if not cls_test_cases.remarks:
                successful_test_result: cls_TestResult = init_test_result(cls_test_cases)

                for test_case_no, _ in enumerate(cls_test_cases.unit_test):
                    test_result = cls_TestResult(test_case_no, cls_test_cases)

                    overall_error_msg, success_unit_test, successful_import_stmt, is_passed = \
                        test_result.process_single_test_case_and_accumulate_results(
                            cls_test_cases, test_case_no, cls_settings, cls_source_code
                        )

                    passed_count += is_passed

                successful_test_result.import_statement = successful_import_stmt
                successful_test_result.unit_test = success_unit_test

                cls_test_cases.remarks = _process_and_save_test_results(
                    source_dir, cls_source_code, cls_settings,
                    successful_test_result, passed_count, overall_error_msg
                )

            test_stats.append({
                "filename": source_code_file,
                "total_test_cases_passed": passed_count,
                "total_test_cases": total_test_case,
                "percentage_passed (%)": (passed_count / total_test_case * 100)
                    if total_test_case > 0 else 0.0,
                "remarks": cls_test_cases.remarks
            })

        test_stats_df = pd.DataFrame(test_stats)
        test_stats_df.index = test_stats_df.index + 1
        logger.info("\n" + tabulate(test_stats_df, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
