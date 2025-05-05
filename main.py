import shutil
import pandas as pd

from typing import NoReturn, List, Optional
from pathlib import Path
from tabulate import tabulate

from TestPilot.common_helpers import cls_Settings, SaveFile, setup_logger
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases
from TestPilot.test_result import cls_TestResult

logger = setup_logger()

def _clean_test_environment(cls_settings:cls_Settings) -> None:
    def _reset_file(file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def _recreate_dir(dir_path: str) -> None:
        path = Path(dir_path)
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    _reset_file(cls_settings.temp_test_file)
    _reset_file(cls_settings.log_file)
    _reset_file(cls_settings.unit_test_file)

    for dir_path in [
        cls_settings.generated_tests_dir,
        cls_settings.finalized_tests_dir,
        cls_settings.failed_tests_dir,
    ]:
        _recreate_dir(dir_path)

def _get_python_files(directory: str) -> List[Path]:
    """
    Recursively retrieves all Python (.py) files within the given directory.

    Args:
        directory (str): The root directory to search for Python files.

    Returns:
        List[Path]: A list of Path objects representing all found .py files.
    """
    return list(Path(directory).rglob("*.py"))


def _run_initial_setup():
    # Read Settings
    cls_settings = cls_Settings()
    # Read Housekeep Prcocessing Folders
    _clean_test_environment(cls_settings)
    return cls_settings


def _process_test_result(
    source_dir: str,
    cls_sourcecode,
    cls_settings,
    success_unit_test: str,
    passed_count: int,
    overall_error_msg: Optional[str] = None
) -> None:
    """
    Processes the test result by saving successful and failed test outputs.

    Args:
        source_dir (str): Source directory path.
        cls_sourcecode: Source code object with source_code_file_path attribute.
        cls_settings: Settings object with finalized_tests_dir and failed_tests_dir attributes.
        success_unit_test (str): The successful unit test code.
        passed_count (int): Number of passed test cases.
        overall_error_msg (Optional[str]): Combined error message from failed tests, if any.
    """
    savefile = SaveFile(source_dir, cls_sourcecode.source_code_file_path)

    if passed_count > 0:
        savefile.save_file(Path(cls_settings.finalized_tests_dir), success_unit_test)
        is_passed, err_msg = cls_TestResult(1, success_unit_test).run_unit_test(cls_settings, cls_sourcecode)
        if not is_passed:
            logger.error(f"Error when compiling code - {err_msg}")

    if overall_error_msg:
        savefile.save_file(
            Path(cls_settings.failed_tests_dir),
            overall_error_msg,
            prefix="err_",
            file_extension=".log"
        )


def main() -> NoReturn:
    cls_settings = _run_initial_setup() 

    logger.info(cls_settings.source_dir_str)
    source_dir_list = [path.strip() for path in cls_settings.source_dir_str.split(",") if path]

    cls_test_cases = cls_Test_Cases()
    for source_dir in source_dir_list:
        source_code_dir = _get_python_files(source_dir)
        passed_count=0
        test_stats=[]
        
        for source_code_file in source_code_dir:
            cls_sourcecode = cls_SourceCode(source_code_file, cls_settings)
            cls_test_cases.process_test_cases(cls_sourcecode, cls_settings)
            success_unit_test=cls_test_cases.import_statement + "\n\n" + \
                cls_test_cases.pytest_fixtures
            idx=0
            overall_error_msg=""
            for idx, test_case in enumerate(cls_test_cases.unit_test, start=1):
                full_test_case = cls_test_cases.import_statement + "\n\n" + \
                    cls_test_cases.pytest_fixtures + "\n\n" + test_case
                cls_test_result_list, error_msg_unit_case = cls_TestResult(idx, full_test_case).\
                    process_test_result(cls_test_cases, cls_settings, cls_sourcecode)
                if cls_test_result_list[-1].is_passed:
                    passed_count+=1
                    success_unit_test+=cls_test_cases.unit_test[-1] + "\n\n"
                else:
                    overall_error_msg+=error_msg_unit_case

            test_stats.append({
                "filename": source_code_file,
                "total_test_cases_passed": passed_count,
                "total_test_cases": idx,
                "percentage_passed (%)": (passed_count / idx * 100) if idx > 0 else 0.0,
                "remarks": cls_test_cases.remarks
            })
            _process_test_result(source_dir, cls_sourcecode, cls_settings,
                                 success_unit_test, passed_count,overall_error_msg)
        test_stats_df = pd.DataFrame(test_stats)
        test_stats_df.index = test_stats_df.index + 1
        logger.info("\n" + tabulate(test_stats_df, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    main()
