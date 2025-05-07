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

    def run_unit_test(self, full_test_case:str, cls_settings:cls_Settings, 
                      cls_source_code:cls_SourceCode) -> Tuple[bool, str]:
        if os.path.exists(cls_settings.temp_test_file):
            os.remove(cls_settings.temp_test_file)

        cls_settings.temp_test_file.write_text(f"# {cls_source_code.source_code_file_path}\n" \
                                               f"{full_test_case}", encoding="utf-8")

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
        err_msg=""
        if not passed:
            err_msg = result.stdout.strip()
        return passed, err_msg

    def _resolve_unit_test_error(self, cls_source_code:cls_SourceCode, 
                                cls_settings:cls_Settings)->str:
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {
            "source_code": cls_source_code.source_code,
            "test_case": self.unit_test,
            "test_case_error": self.error_msg,
            "requirements_txt": cls_source_code.requirements_txt,
            "module_path": cls_source_code.module_path
        }
        full_unit_test = llm_prompt_executor.execute_llm_prompt(cls_settings.llm_resolve_prompt, 
                                                                llm_parameter)
    
        return full_unit_test

    def _print_test_result(self, retry_count: int, 
                          file_path: str) -> str:
        divider = "=" * 80
        section_break = "-" * 80

        output="\n"
        output+=f"\n{divider}\n"
        output+=f"File: {file_path}\n"
        output+=f"TEST CASE {self.test_case_no+1} - Retry {retry_count} - " \
            f"({'Passed' if self.is_passed == True else 'Failed'})\n"
        output+=f"\n"
        output+=f"{section_break}\n"
        output+="Unit Test Code:\n"
        output+=f"{section_break}\n"
        output+=f"\n{self.full_test_case.strip()}\n"
        output+=f"{divider}\n"
        if not self.is_passed:
            output+=f"Error: \n"
            output+=f"{divider}\n"
            output+=f"{self.error_msg}\n"
            output+=f"{divider}\n"
        return output
        
    def _ensure_import_statements(self) -> None:
        import_map = {
            "pytest": "import pytest",
            "patch": "from unittest.mock import patch",
            "mock_open": "from unittest.mock import mock_open",
            "mock": "from unittest import mock",
            "AsyncMock": "from unittest.mock import AsyncMock",
            "MagicMock": "from unittest.mock import MagicMock"
        }
        for keyword, import_line in import_map.items():
            if (keyword in self.unit_test or keyword in self.pytest_fixtures) and \
                import_line not in self.import_statement:
                self.import_statement += f"\n{import_line}"


    def process_test_cases(self, cls_settings:cls_Settings,  
                            cls_source_code:cls_SourceCode)->Tuple[List["cls_TestResult"], Optional[str]]:
        retry_count=0
        test_result_list:List[cls_TestResult]=[]
        overall_error_msg=""
        while retry_count <= cls_settings.max_num_tries and not self.is_passed:
            self._ensure_import_statements()
            self.is_passed, self.error_msg = self.run_unit_test(self.full_test_case, cls_settings, cls_source_code)
            test_report=self._print_test_result(retry_count, cls_source_code.source_code_file_path)
            logger.info(test_report)
            overall_error_msg+=test_report
 
            test_result_list.append(self)

            if not self.is_passed:
                retry_count += 1
                full_unit_test = self._resolve_unit_test_error(cls_source_code, cls_settings)
                llm_prompt_executor = LLMPromptExecutor(cls_settings)
                self._build_unit_test_code(full_unit_test, cls_source_code, cls_settings, 
                                               llm_prompt_executor)
        return test_result_list, overall_error_msg
