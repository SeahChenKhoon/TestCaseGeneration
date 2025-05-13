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
    # Initialisation
    passed_count=0
    test_stats=[]
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
    cls_source_code:cls_SourceCode,
    cls_settings:cls_Settings,
    successful_test_result:cls_TestResult,
    passed_count: int,
    overall_error_msg: Optional[str] = None
) -> str:
    savefile = SaveFile(source_dir, cls_source_code.source_code_file_path)
    is_passed:bool = False
    err_msg:str=""
    if passed_count > 0:
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {"full_unit_tests": successful_test_result.full_test_case}
        successful_test_result.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_settings.llm_organize_imports_prompt, llm_parameter)
        savefile.save_file(Path(cls_settings.finalized_tests_dir), successful_test_result.full_test_case)
        is_passed, err_msg = successful_test_result.run_unit_test(successful_test_result.full_test_case, cls_settings, cls_source_code)

    if overall_error_msg:
        savefile.save_file(
            Path(cls_settings.failed_tests_dir),
            overall_error_msg,
            prefix="err_",
            file_extension=".log"
        )
    if not is_passed:
        return f"Error in compilation - {err_msg}"
    else:
        return f"No Error in compilation"


def init_test_result(cls_test_cases: cls_Test_Cases) -> cls_TestResult:
    test_case = cls_Test_Cases()
    test_case.import_statement = cls_test_cases.import_statement
    test_case.pytest_fixtures = cls_test_cases.pytest_fixtures
    test_case.unit_test = [""]  

    test_result = cls_TestResult(0, test_case)
    return test_result


def main() -> None:
    cls_settings, source_dir_list, cls_test_cases = _run_initial_setup() 
    for source_dir in source_dir_list:
        logger.info(f"Executing directory {source_dir}... ")
        source_code_dir, passed_count, test_stats = _get_python_files(source_dir)
        source_code_file:Path
        for source_code_file in source_code_dir:
            cls_test_cases = cls_Test_Cases()
            
            passed_count, overall_error_msg, successful_import_stmt,\
            success_unit_test, cls_source_code=cls_test_cases.init_variables(source_code_file, \
                                                                             cls_settings)
            total_test_case = cls_test_cases.derive_test_cases(source_dir, cls_source_code, 
                                                               cls_settings)
            if not cls_test_cases.remarks:
                successful_test_result:cls_TestResult=init_test_result(cls_test_cases)
                for test_case_no, _ in enumerate(cls_test_cases.unit_test):
                    test_result=cls_TestResult(test_case_no, cls_test_cases)
                    overall_error_msg, success_unit_test, successful_import_stmt, is_passed=\
                        test_result.process_single_test_case_and_accumulate_results(cls_test_cases,\
                            test_case_no, cls_settings,cls_source_code)
                    passed_count+=is_passed

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
