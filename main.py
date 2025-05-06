import shutil
import pandas as pd

from typing import NoReturn, List, Optional
from pathlib import Path
from tabulate import tabulate

from TestPilot.common_helpers import cls_Settings, SaveFile, LLMPromptExecutor, setup_logger
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


def _process_and_save_test_results(
    source_dir: str,
    cls_source_code:cls_SourceCode,
    cls_settings:cls_Settings,
    successful_test_result:cls_TestResult,
    passed_count: int,
    overall_error_msg: Optional[str] = None
) -> None:
    savefile = SaveFile(source_dir, cls_source_code.source_code_file_path)

    if passed_count > 0:
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {"full_unit_tests": successful_test_result.full_test_case}
        successful_test_result.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_settings.llm_organize_imports_prompt, llm_parameter)
        savefile.save_file(Path(cls_settings.finalized_tests_dir), successful_test_result.full_test_case)
        is_passed, err_msg = successful_test_result.run_unit_test(successful_test_result.full_test_case, cls_settings, cls_source_code)
        if not is_passed:
            return f"Error in compilation - {err_msg}"
        else:
            return f"No Error in compilation"

    if overall_error_msg:
        savefile.save_file(
            Path(cls_settings.failed_tests_dir),
            overall_error_msg,
            prefix="err_",
            file_extension=".log"
        )

def save_initial_test_cases(
    source_dir: str,
    cls_source_code: cls_SourceCode,
    cls_settings: cls_Settings,
    cls_test_cases: cls_Test_Cases
) -> None:
    savefile = SaveFile(source_dir, cls_source_code.source_code_file_path)

    test_case_parts = [
        cls_test_cases.import_statement,
        "",
        cls_test_cases.pytest_fixtures,
        "",
        *cls_test_cases.unit_test
    ]
    
    full_test_case = "\n\n".join(test_case_parts) + "\n\n"
    savefile.save_file(Path(cls_settings.generated_tests_dir), full_test_case,prefix="init_")

def create_successful_test_result(cls_test_cases: cls_Test_Cases) -> cls_TestResult:
    """
    Creates a successful cls_TestResult object based on given cls_Test_Cases.

    Args:
        cls_test_cases (cls_Test_Cases): Source test cases object.

    Returns:
        cls_TestResult: A successful test result object.
    """
    success_test_case = cls_Test_Cases()
    success_test_case.import_statement = cls_test_cases.import_statement
    success_test_case.pytest_fixtures = cls_test_cases.pytest_fixtures
    success_test_case.unit_test = [""]  # One empty unit test as string

    successful_test_result = cls_TestResult(0, success_test_case)
    return successful_test_result


def main() -> NoReturn:
    cls_settings = _run_initial_setup() 
    source_dir_list = [path.strip() for path in cls_settings.source_dir_str.split(",") if path]
    cls_test_cases = cls_Test_Cases()

    for source_dir in source_dir_list:
        logger.info(f"Executing directory {source_dir}... ")
        passed_count=0
        test_stats=[]
        source_code_dir = _get_python_files(source_dir)
        for source_code_file in source_code_dir:
            passed_count=0
            total_test_case=0
            overall_error_msg=""
            successful_import_stmt=""
            success_unit_test=""

            cls_source_code = cls_SourceCode(source_code_file, cls_settings)
            cls_test_cases.derive_test_cases(cls_source_code, cls_settings)
            successful_test_result=create_successful_test_result(cls_test_cases)
            save_initial_test_cases(source_dir, cls_source_code, cls_settings, cls_test_cases)
            for test_case_no, _ in enumerate(cls_test_cases.unit_test):
                test_result_list: List[cls_TestResult]
                error_msg_unit_case: Optional[str]
                total_test_case=len(cls_test_cases.unit_test)
                cls_test_result = cls_TestResult(test_case_no, cls_test_cases)
                test_result_list, error_msg_unit_case = \
                    cls_test_result.process_test_cases(cls_settings, cls_source_code)

                if test_result_list[-1].is_passed:
                    passed_count+=1
                    successful_import_stmt+=test_result_list[-1].import_statement + "\n\n"
                    success_unit_test+=test_result_list[-1].unit_test + "\n\n"
                else:
                    overall_error_msg+=error_msg_unit_case

            successful_test_result.import_statement=successful_import_stmt
            successful_test_result.unit_test=success_unit_test

            cls_test_cases.remarks=_process_and_save_test_results(source_dir, cls_source_code, cls_settings,
                                 successful_test_result, passed_count, overall_error_msg)
            test_stats.append({
                "filename": source_code_file,
                "total_test_cases_passed": passed_count,
                "total_test_cases": total_test_case,
                "percentage_passed (%)": (passed_count / total_test_case * 100) \
                    if total_test_case > 0 else 0.0,
                "remarks": cls_test_cases.remarks
            })
        test_stats_df = pd.DataFrame(test_stats)
        test_stats_df.index = test_stats_df.index + 1
        logger.info("\n" + tabulate(test_stats_df, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    main()
