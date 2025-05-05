import os, re, subprocess
from pathlib import Path
from typing import Tuple, List, NoReturn

from TestPilot.common_helpers import cls_Settings, LLMPromptExecutor
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases
from TestPilot.logger import setup_logger 

logger = setup_logger()

class cls_TestResult:
    def __init__(self, test_case_no:int, unit_test:str):
        self.test_case_no = test_case_no
        self.unit_test = unit_test
        self.is_passed = False
        self.error_msg = None

class cls_TestResult_Set:
    def __init__(self):
        self.results: List[cls_TestResult] = []

    def run_unit_test(self, cls_test_result:cls_TestResult,cls_settings:cls_Settings, 
                      cls_source_code:cls_SourceCode) -> Tuple[bool, str]:
        if os.path.exists(cls_settings.temp_test_file):
            os.remove(cls_settings.temp_test_file)

        cls_settings.temp_test_file.write_text(f"# {cls_source_code.source_code_file_path}\n" \
                                               f"{cls_test_result.unit_test}", encoding="utf-8")

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

    def resolve_unit_test_error(self,  test_case, test_case_error, cls_source_code:cls_SourceCode, 
                                cls_settings:cls_Settings):
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {
            "source_code": cls_source_code.source_code,
            "test_case": full_unit_test,
            "test_case_error": test_case_error,
            "requirements_txt": cls_source_code.requirements_txt,
        }
        full_unit_test = llm_prompt_executor.execute_llm_prompt(cls_Settings.llm_resolve_non_orm_prompt, llm_parameter)

    def add_test_result(self, test_result: cls_TestResult) -> None:
        self.results.append(test_result)

        # self.test_case_no = test_case_no
        # self.unit_test = unit_test
        # self.is_passed = False
        # self.error_msg = None

    def print_test_result(self, cls_test_result:cls_TestResult, retry_count: int, 
                          file_path: str) -> str:
        divider = "=" * 80
        section_break = "-" * 80

        output="\n"
        output+=f"\n{divider}\n"
        output+=f"TEST CASE {cls_test_result.test_case_no} - Retry {retry_count}\n"
        output+=f"File: {file_path}\n"
        output+=f"Status: {'Passed' if cls_test_result.is_passed == True else 'Failed'}\n"
        if not cls_test_result.is_passed:
            output+=f"Error: \n{cls_test_result.error_msg}\n"
        output+=f"{section_break}\n"
        output+="Unit Test Code:\n"
        output+=f"{section_break}\n"
        output+=f"\n{cls_test_result.unit_test.strip()}\n"
        output+=f"{divider}\n"
        return output
        

    def process_test_result(self, cls_test_result:cls_TestResult, cls_settings:cls_Settings,  
                            cls_source_code:cls_SourceCode)->NoReturn:
        retry_count=0

        while retry_count <= cls_settings.max_num_tries and not cls_test_result.is_passed:
            cls_test_result.is_passed, cls_test_result.error_msg = \
                self.run_unit_test(cls_test_result, cls_settings, cls_source_code)
            logger.info(self.print_test_result(cls_test_result, retry_count, 
                                   cls_source_code.source_code_file_path))
            self.add_test_result(cls_test_result)
            retry_count += 1
            if not cls_test_result.is_passed:
                full_unit_test = self.resolve_unit_test_error(cls_source_code.source_code, 
                                                              cls_test_result.unit_test, 
                                                              cls_source_code.source_code, 
                                                              cls_settings)
                
                cls_test_result = cls_TestResult(full_unit_test)
                



                




