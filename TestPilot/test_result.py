import os, re, subprocess
from pathlib import Path
from typing import Tuple, List, NoReturn

from TestPilot.common_helpers import cls_Settings, LLMPromptExecutor, SaveFile, setup_logger
from TestPilot.source_code import cls_SourceCode
from TestPilot.test_cases import cls_Test_Cases

logger = setup_logger()

class cls_TestResult:
    def __init__(self, test_case_no:int, unit_test:str):
        self.test_case_no = test_case_no
        self.unit_test = unit_test
        self.is_passed = False
        self.error_msg = None

    def run_unit_test(self, cls_settings:cls_Settings, 
                      cls_source_code:cls_SourceCode) -> Tuple[bool, str]:
        if os.path.exists(cls_settings.temp_test_file):
            os.remove(cls_settings.temp_test_file)

        cls_settings.temp_test_file.write_text(f"# {cls_source_code.source_code_file_path}\n" \
                                               f"{self.unit_test}", encoding="utf-8")

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
                                cls_settings:cls_Settings):
        llm_prompt_executor = LLMPromptExecutor(cls_settings)
        llm_parameter = {
            "source_code": cls_source_code.source_code,
            "test_case": self.unit_test,
            "test_case_error": self.error_msg,
            "requirements_txt": cls_source_code.requirements_txt,
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
        output+=f"TEST CASE {self.test_case_no} - Retry {retry_count}\n"
        output+=f"File: {file_path}\n"
        output+=f"Status: {'Passed' if self.is_passed == True else 'Failed'}\n"
        if not self.is_passed:
            output+=f"Error: \n{self.error_msg}\n"
        output+=f"{section_break}\n"
        output+="Unit Test Code:\n"
        output+=f"{section_break}\n"
        output+=f"\n{self.unit_test.strip()}\n"
        output+=f"{divider}\n"
        return output
        

    def process_test_result(self, cls_test_cases:cls_Test_Cases,cls_settings:cls_Settings,  
                            cls_source_code:cls_SourceCode)->NoReturn:
        retry_count=0
        test_result_list:List[cls_TestResult]=[]
        overall_error_msg=""
        while retry_count <= cls_settings.max_num_tries and not self.is_passed:
            self.is_passed, self.error_msg = \
                self.run_unit_test(cls_settings, cls_source_code)

            test_report=self._print_test_result(retry_count, cls_source_code.source_code_file_path)
            logger.info(test_report)
            overall_error_msg+=test_report
            test_result_list.append(self)
            
            retry_count += 1
            if not self.is_passed:
                full_unit_test = self._resolve_unit_test_error(cls_source_code, cls_settings)
                llm_prompt_executor = LLMPromptExecutor(cls_settings)
                test_case = cls_test_cases._extract_test_case_from_test_cases(llm_prompt_executor, 
                                                    cls_settings.llm_extract_test_cases_prompt, 
                                                        full_unit_test, True)
                cls_test_cases.unit_test[self.test_case_no-1]=test_case
                full_unit_test = cls_test_cases.import_statement + "\n\n" + \
                    cls_test_cases.pytest_fixtures + "\n\n" + test_case

                self.__init__(self.test_case_no, full_unit_test)
        return test_result_list, overall_error_msg





                




