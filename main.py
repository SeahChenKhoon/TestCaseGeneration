import shutil
import sys

from typing import NoReturn, Tuple, List
from pathlib import Path

from TestPilot.utils.logger import setup_logger 
from TestPilot.models.settings import Settings

logger = setup_logger()

def clean_test_environment(settings) -> None:
    """
    Cleans and prepares the test environment by:
    - Resetting temp test and log files
    - Recreating directories for generated, finalized, and failed test files

    Args:
        settings: An object containing paths such as temp_test_file, log_file, 
                  generated_tests_dir, finalized_tests_dir, and failed_tests_dir.
    """
    def _reset_file(file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def _recreate_dir(dir_path: str) -> None:
        path = Path(dir_path)
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    _reset_file(settings.temp_test_file)
    _reset_file(settings.log_file)

    for dir_path in [
        settings.generated_tests_dir,
        settings.finalized_tests_dir,
        settings.failed_tests_dir,
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


def run_initial_setup() -> Tuple['Settings', List[Path]]:
    """
    Executes initial setup steps before processing, including:
    - Loading environment settings
    - Cleaning up the test environment
    - Scanning the source directory for Python files

    Returns:
        Tuple[Settings, List[Path]]: 
            - An instance of the Settings class containing environment configuration.
            - A list of Python source code files found in the configured source directory.
    """
    logger.info(f"run_initial_setup start")
    # Read Settings
    settings = Settings()
    # Read Housekeep Prcocessing Folders
    clean_test_environment(settings)
    logger.info(f"run_initial_setup end")
    return settings

from TestPilot.biz.source_file_biz import SourceCodeBiz
def main() -> NoReturn:
    settings = run_initial_setup() 

    test_stats = []
    logger.info(settings.source_dir_str)
    source_dir_list = [path.strip() for path in settings.source_dir_str.split(",") if path]
    # source_dir_list = [Path(path.strip()) for path in settings.source_dir_str.split(",") if path.strip()]
    for source_dir in source_dir_list:
        source_code_files = _get_python_files(source_dir)
        for source_code_file in source_code_files:
            source_code_file = SourceCodeBiz.process_source_file(source_code_file, settings)
            logger.info(f"Hello World - Process Unit Test Case")
    logger.info(f"Produce Report")


    # stats_df, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    main()
