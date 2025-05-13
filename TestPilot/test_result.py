import os, re, subprocess
from pathlib import Path
from typing import Tuple, List, Optional

from TestPilot.common_helpers import cls_Settings, LLMPromptExecutor, SaveFile, setup_logger
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases


logger = setup_logger()

class cls_TestResult(cls_Test_Cases):
    def __init__(self, test_case_no:int, cls_test_cases:cls_Test_Cases):
        self.import_statement:str = cls_test_cases.import_statement
        self.pytest_fixtures:str = cls_test_cases.pytest_fixtures
        self.test_case_no:int = test_case_no
        self.unit_test:str = cls_test_cases.unit_test[test_case_no]
        self.is_passed:bool = False
        self.error_msg:str = None
    @property
    def full_test_case(self) -> str:
        return f"{self.import_statement}\n\n{self.pytest_fixtures}\n\n{self.unit_test}"

    def run_unit_test(
        self,
        full_test_case: str,
        cls_settings: cls_Settings,
        cls_source_code: cls_SourceCode
    ) -> Tuple[bool, str]:
        """
        Writes the full unit test case to a temporary test file and executes it using pytest.
        Captures the result and returns whether the test passed and any error message.

        Args:
            full_test_case (str): The complete unit test code to execute.
            cls_settings (cls_Settings): Settings object containing paths and configs.
            cls_source_code (cls_SourceCode): Source code metadata used for logging context.

        Returns:
            Tuple[bool, str]: A tuple where:
                - The first element indicates whether the test passed (True/False).
                - The second element contains the error message if failed, otherwise an empty string.
        """
        if os.path.exists(cls_settings.temp_test_file):
            os.remove(cls_settings.temp_test_file)

        Path(cls_settings.temp_test_file).write_text(
            f"# {cls_source_code.source_code_file_path}\n{full_test_case}",
            encoding="utf-8"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(".").resolve())

        result = subprocess.run(
            ["pytest", str(cls_settings.temp_test_file), "--tb=short", "--quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        passed = result.returncode == 0
        err_msg = result.stdout.strip() if not passed else ""

        return passed, err_msg


    def _resolve_unit_test_error(
        self,
        cls_source_code: cls_SourceCode,
        cls_settings: cls_Settings
    ) -> str:
        """
        Uses the LLM to resolve a previously failed unit test by providing the source code,
        original test, error message, and environment context.

        Args:
            cls_source_code (cls_SourceCode): Object containing source code, module path, and requirements.
            cls_settings (cls_Settings): Configuration and prompt templates used for LLM interaction.

        Returns:
            str: A corrected unit test case returned by the LLM.
        """
        llm_prompt_executor = LLMPromptExecutor(cls_settings)

        llm_parameter = {
            "source_code": cls_source_code.source_code,
            "test_case": self.unit_test,
            "test_case_error": self.error_msg,
            "requirements_txt": cls_source_code.requirements_txt,
            "module_path": cls_source_code.module_path
        }

        full_unit_test = llm_prompt_executor.execute_llm_prompt(
            cls_settings.llm_resolve_prompt,
            llm_parameter
        )

        return full_unit_test


    def _print_test_result(
        self,
        retry_count: int,
        file_path: str
    ) -> str:
        """
        Generates a formatted string summarizing the test case result, including:
        - File path
        - Retry count
        - Pass/fail status
        - Unit test code
        - Error message (if failed)

        Args:
            retry_count (int): The number of retry attempts made for this test case.
            file_path (str): The path to the source file being tested.

        Returns:
            str: A formatted string summarizing the test result.
        """
        divider = "=" * 80
        section_break = "-" * 80

        output = "\n"
        output += f"\n{divider}\n"
        output += f"File: {file_path}\n"
        output += f"TEST CASE {self.test_case_no + 1} - Retry {retry_count} - " \
                f"({'Passed' if self.is_passed else 'Failed'})\n"
        output += f"\n{section_break}\n"
        output += "Unit Test Code:\n"
        output += f"{section_break}\n"
        output += f"\n{self.full_test_case.strip()}\n"

        if not self.is_passed:
            output += f"\n{section_break}\n"
            output += "Error: \n"
            output += f"{section_break}\n"
            output += f"{self.error_msg}\n"

        output += f"{divider}\n"
        return output
        

    def _ensure_import_statements(self) -> None:
        """
        Ensures all required import statements are present based on keywords used
        in the unit test body and pytest fixtures.

        If a keyword (e.g., 'patch', 'MagicMock') is found in the test code or fixture block,
        and its corresponding import line is not already included, it appends the necessary
        import statement to `self.import_statement`.
        """
        import_map = {
            "pytest": "import pytest",
            "patch": "from unittest.mock import patch",
            "mock_open": "from unittest.mock import mock_open",
            "mock": "from unittest import mock",
            "AsyncMock": "from unittest.mock import AsyncMock",
            "MagicMock": "from unittest.mock import MagicMock",
        }

        for keyword, import_line in import_map.items():
            if (
                keyword in self.unit_test or keyword in self.pytest_fixtures
            ) and import_line not in self.import_statement:
                self.import_statement += f"\n{import_line}"


    def process_single_test_case_and_accumulate_results(
        self,
        cls_test_cases: cls_Test_Cases,
        test_case_no: int,
        cls_settings: cls_Settings,
        cls_source_code: cls_SourceCode
    ) -> Tuple[str, str, str, int]:
        """
        Processes a single test case, evaluates its result, and accumulates relevant outputs.

        Args:
            cls_test_cases (cls_Test_Cases): The container holding import statements and unit tests.
            test_case_no (int): The index of the current test case being processed.
            cls_settings (cls_Settings): Settings object used during test execution.
            cls_source_code (cls_SourceCode): Object containing the source code being tested.

        Returns:
            Tuple[str, str, str, int]: A tuple containing:
                - overall_error_msg (str): Collected error message if the test failed.
                - success_unit_test (str): Generated unit test code if passed.
                - successful_import_stmt (str): Relevant import statements if passed.
                - is_passed (int): 1 if the test passed, 0 otherwise.
        """
        is_passed = 0
        overall_error_msg = ""
        successful_import_stmt = ""
        success_unit_test = ""

        cls_test_result = cls_TestResult(test_case_no, cls_test_cases)
        test_result_list, error_msg_unit_case = cls_test_result._process_test_cases(
            cls_settings, cls_source_code
        )

        if test_result_list[-1].is_passed:
            is_passed = 1
            successful_import_stmt += test_result_list[-1].import_statement + "\n\n"
            success_unit_test += test_result_list[-1].unit_test + "\n\n"
        else:
            overall_error_msg += error_msg_unit_case or ""

        return overall_error_msg, success_unit_test, successful_import_stmt, is_passed


    def _process_test_cases(
        self,
        cls_settings: cls_Settings,
        cls_source_code: cls_SourceCode
    ) -> Tuple[List["cls_TestResult"], Optional[str]]:
        """
        Attempts to execute and resolve a test case multiple times until it passes 
        or the maximum retry limit is reached.

        Each retry involves patching import statements, running the test, and logging 
        the result. If the test fails, the function attempts to resolve the error using 
        LLM-based refinement and rebuilds the test case.

        Args:
            cls_settings (cls_Settings): Settings including retry limit and test configuration.
            cls_source_code (cls_SourceCode): The source code file under test.

        Returns:
            Tuple[List[cls_TestResult], Optional[str]]:
                - A list of test result snapshots (1 per attempt).
                - A string containing the combined error messages, if any.
        """
        retry_count = 0
        test_result_list: List[cls_TestResult] = []
        overall_error_msg = ""

        while retry_count <= cls_settings.max_num_tries and not self.is_passed:
            self._ensure_import_statements()

            self.is_passed, self.error_msg = self.run_unit_test(
                self.full_test_case, cls_settings, cls_source_code
            )

            test_report = self._print_test_result(retry_count, cls_source_code.source_code_file_path)
            logger.info(test_report)
            overall_error_msg += test_report

            test_result_list.append(self)

            if not self.is_passed:
                retry_count += 1
                full_unit_test = self._resolve_unit_test_error(cls_source_code, cls_settings)
                llm_prompt_executor = LLMPromptExecutor(cls_settings)
                self._build_unit_test_code(
                    full_unit_test, cls_source_code, cls_settings, llm_prompt_executor
                )

        return test_result_list, overall_error_msg
