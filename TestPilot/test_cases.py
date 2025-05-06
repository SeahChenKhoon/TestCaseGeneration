from typing import List, Optional, NoReturn, Tuple, Union
from pathlib import Path
import re, os

from TestPilot.source_code import cls_SourceCode
from TestPilot.common_helpers import cls_Settings, LLMPromptExecutor, setup_logger

logger = setup_logger()

class cls_Test_Cases:
    def __init__(self):
        self.import_statement:str = ""
        self.pytest_fixtures:str = ""
        self.unit_test = []
        self.remarks:str=""

    def _should_generate_unit_test(self,code: str) -> bool:
        """
        Determines if a unit test should be generated based on the structure of the code.
        
        Returns:
            bool: True if the code is meaningful for unit testing, False otherwise.
        """

        # Check if it's only simple imports (pure import file)
        only_imports_pattern = re.compile(
            r'^\s*(from\s+\.\s+import\s+\w+|import\s+\w+)(\s*#.*)?$', re.MULTILINE
        )
        if only_imports_pattern.findall(code):
            non_import_lines = [
                line for line in code.splitlines() 
                if line.strip() and not re.match(r'^(from\s+\.\s+import\s+\w+|import\s+\w+)', 
                                                 line.strip())
            ]
            if not non_import_lines:
                return False  # Only imports, no logic

        # Check if it defines only a BaseSettings subclass with no custom logic
        if "class" in code and "BaseSettings" in code:
            # Rough check: if it's only class definition + attribute definitions
            has_methods = re.search(r'def\s+\w+', code)
            if not has_methods:
                return False  # Just settings, no business logic

        # Check if it defines only SQLAlchemy ORM models
        if "declarative_base" in code and "Column(" in code:
            # Again: if there are no custom methods inside the models
            has_methods = re.search(r'def\s+\w+', code)
            if not has_methods:
                return False  # Only ORM model definition, no logic

        # Otherwise, yes, meaningful code — generate unit test
        return True

    def _convert_relative_to_absolute_imports(self, code: str, file_path: str) -> str:
        file_path_obj = Path(file_path).with_suffix("")
        module_parts = list(file_path_obj.parts)
        pattern = re.compile(r"from\s+(\.+)([\w\.]*)\s+import\s+(\w+)")

        def replacer(match):
            dots = match.group(1)                  # ., .., ...
            relative_module = match.group(2)       # can be empty or nested (e.g., llm_handler or sub.module)
            imported_name = match.group(3)         # imported object

            levels_up = len(dots)
            if levels_up > len(module_parts):
                raise ValueError(f"Too many dots in relative import for path {file_path}")

            # Trim path upward based on levels
            base_parts = module_parts[:len(module_parts) - levels_up]
            # Append relative module parts (if any)
            if relative_module:
                base_parts.extend(relative_module.split("."))

            absolute_import = ".".join(base_parts)
            return f"from {absolute_import} import {imported_name}"
        return pattern.sub(replacer, code)

    def _extract_function_class_and_factory_assignments(self, code: str) -> List[str]:
        # Match top-level (not indented) functions
        function_names = re.findall(
            r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code, re.MULTILINE
        )

        # Match class names
        class_names = re.findall(
            r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', code, re.MULTILINE
        )

        factory_assignments = re.findall(
            r'^([A-Z][a-zA-Z0-9_]*)\s*=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(', code, re.MULTILINE
        )

        return sorted(set(function_names + class_names + factory_assignments))

    def _construct_module_import(self, function_names: List[str], source_code_path: str) -> str:
        """
        Generates a dynamic import statement based on the provided function names
        and source file path.

        Args:
            function_names (List[str]): Names of functions to import.
            source_code_path (str): Source file path (e.g., 'theory_evaluation/llm_utils.py').

        Returns:
            str: Formatted import statement (e.g., 'from theory_evaluation.llm_utils import func1, func2').
        """

        # Convert to dotted module path and remove .py extension
        module_path = os.path.splitext(source_code_path)[0].replace(os.sep, ".")

        names_str = ", ".join(function_names)
        import_line = f"from {module_path} import {names_str}"

        return import_line


    def _derive_import_statements(self, llm_prompt_executor:LLMPromptExecutor, 
                                  cls_setting:cls_Settings, cls_source_code:cls_SourceCode, 
                                  generated_unit_test_code:str):

        # Derive import statement from test_case
        llm_parameter = {"source_code": generated_unit_test_code}
        unit_test_import_statements = llm_prompt_executor.execute_llm_prompt(
            cls_setting.llm_extract_import_prompt, llm_parameter)
        
        if self.__class__.__name__ == "cls_Test_Cases":
            # Derive import statement from source code
            llm_parameter = {"source_code": cls_source_code.source_code}
            source_code_import_statements = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_extract_import_prompt, llm_parameter)
            source_code_import_statements = self._convert_relative_to_absolute_imports(
                source_code_import_statements, cls_source_code.source_code_file_path)
            function_names = self._extract_function_class_and_factory_assignments(
                cls_source_code.source_code)
            dynamic_imports = self._construct_module_import(function_names, 
                                                        cls_source_code.source_code_file_path)
            source_code_import_statements += "\n" + dynamic_imports + "\n"

            # Merge the source code and unit test import statements together
            llm_parameter = {"source_code_import_statements": source_code_import_statements,
                            "unit_test_import_statements": unit_test_import_statements}
            self.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_merge_imports_prompt, llm_parameter)
        else:
            # Merge the existing and unit test import statements together
            llm_parameter = {"existing_import_statements": self.import_statement,
                            "unit_test_import_statements": unit_test_import_statements}
            self.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_merge_unittest_existing_imports_prompt, llm_parameter)            


    def _generates_test_cases(self, cls_source_code:cls_SourceCode, cls_setting:cls_Settings, 
                              llm_prompt_executor)->str:
        if cls_setting.should_generate_tests:
            generated_unit_test_code= Path(cls_setting.unit_test_file).read_text(encoding="utf-8")
        else:
            llm_parameter = {"python_version": cls_source_code.python_version, 
                            "requirements_txt": cls_source_code.requirements_txt,
                            "file_content": cls_source_code.source_code,
                            "module_path": cls_source_code.module_path
                            }
            generated_unit_test_code = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_generate_unit_tests_prompt, llm_parameter)
            generated_unit_test_code = self._convert_relative_to_absolute_imports(
                generated_unit_test_code, cls_source_code.source_code_file_path)
            generated_unit_test_code = self._rewrite_module_references(
                cls_source_code.source_code_file_path,
                generated_unit_test_code)
        return generated_unit_test_code

    def _derive_pytest_fixture(self, generated_unit_test_code:str, cls_setting:cls_Settings,
                              llm_prompt_executor:LLMPromptExecutor):
        llm_parameter={"unit_test_code": generated_unit_test_code}
        pytest_fixtures = llm_prompt_executor.execute_llm_prompt(
            cls_setting.llm_extract_pytest_fixture_prompt, llm_parameter)
        if pytest_fixtures:
            self.import_statement+="\nimport pytest"
        return pytest_fixtures

    def _extract_test_functions(self,code: str) -> List[str]:
        """
        Extracts all test functions (including decorators like @pytest.mark.asyncio)
        from the provided Python source code and returns them as a list of strings.
        """
        # Pattern to match each @pytest.mark.asyncio async def test_xxx(...) with its full body
        pattern = re.compile(
            r'(@pytest\.mark\.asyncio\s*\n)?'                      # Optional @pytest.mark.asyncio
            r'(?:async\s+)?def\s+test_[\w_]+\s*\([^)]*\):'  # Function definition
            r'(?:\n(?: {4}|\t).+)+',                               # Indented function body
            re.MULTILINE
        )

        return [match.group().strip() for match in pattern.finditer(code)]


    def _extract_test_case_from_test_cases(
        self, llm_prompt_executor,
        prompt,
        generated_unit_test_code: str
    )->Union[str, List[str]]:
        llm_parameter = {"unit_test_code": generated_unit_test_code}
        all_test_cases_str = llm_prompt_executor.execute_llm_prompt(
            prompt,
            llm_parameter
        )
         
        all_test_cases_list = self._extract_test_functions(all_test_cases_str)
        if self.__class__.__name__ == "cls_Test_Cases":
            self.unit_test=all_test_cases_list
        else:
            self.unit_test=all_test_cases_list[0]
    
    def _build_unit_test_code(self, generated_unit_test_code: str, 
                             cls_source_code: cls_SourceCode, cls_setting:cls_Settings, 
                             llm_prompt_executor:LLMPromptExecutor) -> Tuple[str]:
        self._derive_import_statements(llm_prompt_executor, cls_setting, 
                                                          cls_source_code, generated_unit_test_code)
        pytest_fixtures = self._derive_pytest_fixture(generated_unit_test_code,cls_setting, llm_prompt_executor)
        self._extract_test_case_from_test_cases(llm_prompt_executor, 
                                               cls_setting.llm_extract_test_cases_prompt, 
                                                generated_unit_test_code)
        return pytest_fixtures

    def _rewrite_module_references(self, source_code_path: str, source_code: str) -> str:
        """
        Replaces all import-like references to the base module name (inferred from the filename)
        with its fully qualified module path, while avoiding redundant prefixes like
        'pkg.pkg.module' becoming just 'pkg.module'.

        Args:
            source_code_path (str): e.g., 'theory_evaluation/llm_handler.py'
            source_code (str): Original source code string

        Returns:
            str: Updated source code with module name replaced by full dotted module path
        """
        path = Path(source_code_path)
        if path.suffix == ".py":
            path = path.with_suffix("")

        module_path = ".".join(path.parts)  # e.g., 'theory_evaluation.llm_handler'
        base_name = path.name              # e.g., 'llm_handler'
        top_level_pkg = path.parts[0]      # e.g., 'theory_evaluation'

        # Replace only word-boundary matches to avoid false positives
        pattern = r'\b' + re.escape(base_name) + r'\b'
        rewritten_code = re.sub(pattern, module_path, source_code)

        # Clean up any redundant repeated package names, e.g., 'pkg.pkg.' => 'pkg.'
        repeated_prefix = f"{top_level_pkg}.{top_level_pkg}."
        cleaned_code = rewritten_code.replace(repeated_prefix, f"{top_level_pkg}.")

        return cleaned_code

    def derive_test_cases(self, cls_source_code:cls_SourceCode, cls_settings:cls_Settings) -> None:
        remarks=""
        if self._should_generate_unit_test(cls_source_code.source_code):
            llm_prompt_executor = LLMPromptExecutor(cls_settings)
            # Generates Test Cases
            generated_unit_test_code = self._generates_test_cases(cls_source_code, cls_settings, 
                                                                  llm_prompt_executor)
            # Build unit test component
            pytest_fixtures = self._build_unit_test_code(generated_unit_test_code,
                                                            cls_source_code, cls_settings, 
                                                            llm_prompt_executor)
            self.pytest_fixtures=pytest_fixtures
            self.remarks = ""
        else:
            self.remarks = "Skipped (config/models/imports only)"
            logger.warning(f"Skipping unit test generation for " 
                           f"{cls_source_code.source_code_file_path}" 
                           " as it only contains configuration, models, or pure imports.")