import shutil
import sys

from typing import NoReturn, Tuple, List
from pathlib import Path

from TestPilot.logger import setup_logger 
from TestPilot.common_helpers import cls_Settings
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases

logger = setup_logger()

def clean_test_environment(cls_settings:cls_Settings) -> None:
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


def run_initial_setup():
    logger.info(f"run_initial_setup start")
    # Read Settings
    cls_settings = cls_Settings()
    # Read Housekeep Prcocessing Folders
    clean_test_environment(cls_settings)
    logger.info(f"run_initial_setup end")
    return cls_settings

from TestPilot.source_code import cls_SourceCode

def main() -> NoReturn:
    cls_settings = run_initial_setup() 

    test_stats = []
    logger.info(cls_settings.source_dir_str)
    source_dir_list = [path.strip() for path in cls_settings.source_dir_str.split(",") if path]

    cls_test_cases = cls_Test_Cases()    
    for source_dir in source_dir_list:
        source_code_dir = _get_python_files(source_dir)
        for source_code_file in source_code_dir:
            cls_sourcecode = cls_SourceCode(source_code_file, cls_settings)
            cls_sourcecode.process_source_file()
            cls_test_cases.process_test_cases(cls_sourcecode, cls_settings)
    logger.info(f"Produce Report")

if __name__ == "__main__":
    main()
