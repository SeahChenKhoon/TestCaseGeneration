from typing import List, Optional, NoReturn, Tuple, Union
from pathlib import Path
import re, os

from TestPilot.source_code import cls_SourceCode
from TestPilot.common_helpers import cls_Settings, LLMPromptExecutor, setup_logger, SaveFile

logger = setup_logger()

class cls_Test_Cases:
    def __init__(self):
        self.import_statement:str = ""
        self.pytest_fixtures:str = ""
        self.unit_test: List[str] = []
        self.remarks:str=""


    def _should_generate_unit_test(self, code: str) -> bool:
        """
        Determines whether a unit test should be generated for the given source code.

        The function avoids generating tests for files that contain only:
        - Pure import statements with no logic,
        - Pydantic BaseSettings subclass definitions without methods,
        - SQLAlchemy ORM model definitions without any custom logic.

        Args:
            code (str): The full source code to analyze.

        Returns:
            bool: True if the code contains meaningful logic and is suitable for unit test generation,
                False if it's just config-like or structural code with no testable behavior.
        """

        # Check if it's only simple imports (pure import file)
        only_imports_pattern = re.compile(
            r'^\s*(from\s+\.\s+import\s+\w+|import\s+\w+)(\s*#.*)?$', re.MULTILINE
        )
        if only_imports_pattern.findall(code):
            non_import_lines = [
                line for line in code.splitlines()
                if line.strip() and not re.match(r'^(from\s+\.\s+import\s+\w+|import\s+\w+)', line.strip())
            ]
            if not non_import_lines:
                return False  # Only imports, no logic

        # Check if it defines only a BaseSettings subclass with no custom logic
        if "class" in code and "BaseSettings" in code:
            has_methods = re.search(r'def\s+\w+', code)
            if not has_methods:
                return False  # Just settings, no business logic

        # Check if it defines only SQLAlchemy ORM models
        if "declarative_base" in code and "Column(" in code:
            has_methods = re.search(r'def\s+\w+', code)
            if not has_methods:
                return False  # Only ORM model definitions, no logic

        # Otherwise, it's meaningful code
        return True


    def _convert_relative_to_absolute_imports(self, code: str, file_path: str) -> str:
        """
        Converts all relative import statements (e.g., `from . import x`, `from ..module import y`)
        in the given code string to absolute imports based on the provided file path.

        This is useful to ensure compatibility during standalone test execution, where relative imports
        may fail due to lack of package context.

        Args:
            code (str): The source code containing potential relative import statements.
            file_path (str): The file path used to resolve the module’s absolute import path.

        Returns:
            str: Modified source code with all relative imports rewritten as absolute imports.

        Raises:
            ValueError: If the number of relative levels exceeds the depth of the file path.
        """
        file_path_obj = Path(file_path).with_suffix("")  # Remove `.py` extension
        module_parts = list(file_path_obj.parts)

        pattern = re.compile(r"from\s+(\.+)([\w\.]*)\s+import\s+(\w+)")

        def replacer(match: re.Match) -> str:
            dots = match.group(1)                 # e.g., ., .., ...
            relative_module = match.group(2)      # e.g., "", "llm_handler", "sub.module"
            imported_name = match.group(3)        # e.g., "OpenAI_llm"

            levels_up = len(dots)
            if levels_up > len(module_parts):
                raise ValueError(f"Too many dots in relative import for path: {file_path}")

            base_parts = module_parts[:len(module_parts) - levels_up]
            if relative_module:
                base_parts.extend(relative_module.split("."))

            absolute_import = ".".join(base_parts)
            return f"from {absolute_import} import {imported_name}"

        return pattern.sub(replacer, code)


    def _extract_function_class_and_factory_assignments(self, code: str) -> List[str]:
        """
        Extracts top-level function names, class names, and factory-style assignments
        from the provided source code.

        It captures:
        - `def` and `async def` function definitions
        - `class` definitions
        - Assignments that follow the pattern: `PascalCaseName = SomeFactory(...)`

        Args:
            code (str): The raw Python source code as a string.

        Returns:
            List[str]: A sorted list of unique identifiers found (function names, class names, 
                    and assignment targets).
        """
        # Match top-level (non-indented) function definitions
        function_names = re.findall(
            r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code, re.MULTILINE
        )

        # Match top-level class definitions
        class_names = re.findall(
            r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', code, re.MULTILINE
        )

        # Match top-level PascalCase assignments (e.g., Config = Settings(...))
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


    def _derive_import_statements(
        self,
        llm_prompt_executor: LLMPromptExecutor,
        cls_setting: cls_Settings,
        cls_source_code: cls_SourceCode,
        generated_unit_test_code: str
    ) -> None:
        """
        Derives and sets the final import statements to be included in the unit test.

        Depending on the context (e.g., cls_Test_Cases vs. other class), this method:
        - Extracts imports from the generated unit test code.
        - Optionally also extracts and rewrites imports from the source code (if inside cls_Test_Cases).
        - Merges the imports using the appropriate LLM prompt.

        Args:
            llm_prompt_executor (LLMPromptExecutor): LLM client wrapper used for executing prompts.
            cls_setting (cls_Settings): Settings including prompts for extraction and merging.
            cls_source_code (cls_SourceCode): The original source code object.
            generated_unit_test_code (str): The generated unit test code to analyze.
        """
        # Step 1: Extract imports from the generated unit test
        llm_parameter = {"source_code": generated_unit_test_code}
        unit_test_import_statements = llm_prompt_executor.execute_llm_prompt(
            cls_setting.llm_extract_import_prompt,
            llm_parameter
        )

        if self.__class__.__name__ == "cls_Test_Cases":
            # Step 2: Extract and convert source code imports
            llm_parameter = {"source_code": cls_source_code.source_code}
            source_code_import_statements = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_extract_import_prompt,
                llm_parameter
            )

            source_code_import_statements = self._convert_relative_to_absolute_imports(
                source_code_import_statements,
                cls_source_code.source_code_file_path
            )

            # Step 3: Merge source code and unit test imports
            llm_parameter = {
                "source_code_import_statements": source_code_import_statements,
                "unit_test_import_statements": unit_test_import_statements
            }
            self.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_merge_imports_prompt,
                llm_parameter
            )
        else:
            # Merge unit test imports with existing imports
            llm_parameter = {
                "existing_import_statements": self.import_statement,
                "unit_test_import_statements": unit_test_import_statements
            }
            self.import_statement = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_merge_unittest_existing_imports_prompt,
                llm_parameter
            )         


    def _generates_test_cases(
        self,
        cls_source_code: cls_SourceCode,
        cls_setting: cls_Settings,
        llm_prompt_executor: LLMPromptExecutor
    ) -> str:
        """
        Generates unit test code using the LLM, or loads it from an existing file
        if test generation is disabled in the settings.

        Args:
            cls_source_code (cls_SourceCode): Object containing source code, Python version,
                                            requirements, and module path.
            cls_setting (cls_Settings): Configuration object containing test flags and file paths.
            llm_prompt_executor (LLMPromptExecutor): LLM interface for executing prompts.

        Returns:
            str: The generated or loaded unit test code as a string.
        """
        if cls_setting.should_generate_tests:
            generated_unit_test_code = Path(cls_setting.unit_test_file).read_text(encoding="utf-8")
        else:
            llm_parameter = {
                "python_version": cls_source_code.python_version,
                "requirements_txt": cls_source_code.requirements_txt,
                "file_content": cls_source_code.source_code,
                "module_path": cls_source_code.module_path
            }
            generated_unit_test_code = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_generate_unit_tests_prompt,
                llm_parameter
            )

        return generated_unit_test_code


    def _derive_pytest_fixture(
        self,
        generated_unit_test_code: str,
        cls_setting: cls_Settings,
        llm_prompt_executor: LLMPromptExecutor
    ) -> str:
        """
        Extracts pytest fixtures from the generated unit test code using the LLM.
        If any fixtures are found, ensures the `import pytest` statement is added.

        Args:
            generated_unit_test_code (str): The full unit test code generated by the LLM.
            cls_setting (cls_Settings): Settings object including the fixture extraction prompt.
            llm_prompt_executor (LLMPromptExecutor): Executor that sends prompts to the LLM.

        Returns:
            str: The extracted pytest fixture code (may be empty if none found).
        """
        llm_parameter = {"unit_test_code": generated_unit_test_code}
        pytest_fixtures = llm_prompt_executor.execute_llm_prompt(
            cls_setting.llm_extract_pytest_fixture_prompt,
            llm_parameter
        )

        if pytest_fixtures:
            self.import_statement += "\nimport pytest"

        return pytest_fixtures


    def _extract_test_functions(self, code: str) -> List[str]:
        """
        Extracts all top-level pytest-style test functions from the provided source code.

        Specifically, it matches:
        - Synchronous and asynchronous test functions (`def test_xxx(...)` and `async def test_xxx(...)`)
        - Functions optionally decorated with `@pytest.mark.asyncio`
        - Entire function bodies (based on consistent indentation)

        Args:
            code (str): Python source code containing one or more test functions.

        Returns:
            List[str]: A list of matched test functions as complete strings, including decorators.
        """
        pattern = re.compile(
            r'(@pytest\.mark\.asyncio\s*\n)?'               # Optional decorator
            r'(?:async\s+)?def\s+test_[\w_]+\s*\([^)]*\):'  # Function definition
            r'(?:\n(?: {4}|\t).+)+',                        # Indented function body
            re.MULTILINE
        )

        return [match.group().strip() for match in pattern.finditer(code)]


    def _extract_test_case_from_test_cases(
        self,
        llm_prompt_executor: LLMPromptExecutor,
        prompt: str,
        generated_unit_test_code: str
    ) -> None:
        """
        Extracts test case(s) from the LLM-generated unit test code using a given prompt.

        The method:
        - Sends the generated unit test code to the LLM to extract distinct test cases.
        - Parses the returned string to separate test functions.
        - Stores all extracted test functions if the current class is `cls_Test_Cases`,
        otherwise stores only the first one.

        Args:
            llm_prompt_executor (LLMPromptExecutor): Executor that sends prompts to the LLM.
            prompt (str): The LLM prompt used to extract individual test cases.
            generated_unit_test_code (str): The raw unit test code from which test cases are extracted.

        Returns:
            None
        """
        llm_parameter = {"unit_test_code": generated_unit_test_code}

        all_test_cases_str = llm_prompt_executor.execute_llm_prompt(
            prompt,
            llm_parameter
        )

        all_test_cases_list = self._extract_test_functions(all_test_cases_str)

        if self.__class__.__name__ == "cls_Test_Cases":
            self.unit_test = all_test_cases_list
        else:
            self.unit_test = all_test_cases_list[0] if all_test_cases_list else ""
            
    
    def _build_unit_test_code(
        self,
        generated_unit_test_code: str,
        cls_source_code: cls_SourceCode,
        cls_setting: cls_Settings,
        llm_prompt_executor: LLMPromptExecutor
    ) -> None:
        """
        Builds the unit test code components by:
        - Deriving and merging import statements.
        - Extracting and merging pytest fixtures.
        - Extracting individual test cases from the full unit test block.

        Args:
            generated_unit_test_code (str): The full unit test code generated by the LLM.
            cls_source_code (cls_SourceCode): Object representing the original source code.
            cls_setting (cls_Settings): Test configuration and LLM prompts.
            llm_prompt_executor (LLMPromptExecutor): LLM client used for prompt execution.

        Returns:
            None
        """
        # Step 1: Derive and merge import statements
        self._derive_import_statements(
            llm_prompt_executor, cls_setting, cls_source_code, generated_unit_test_code
        )

        # Step 2: Extract and merge pytest fixtures
        pytest_fixtures = self._derive_pytest_fixture(
            generated_unit_test_code, cls_setting, llm_prompt_executor
        )

        if self.pytest_fixtures and pytest_fixtures:
            llm_parameter = {
                "initial_pytest_fixtures": self.pytest_fixtures,
                "new_pytest_fixtures": pytest_fixtures
            }
            self.pytest_fixtures = llm_prompt_executor.execute_llm_prompt(
                cls_setting.llm_merge_pytest_fixtures_prompt,
                llm_parameter
            )
        else:
            self.pytest_fixtures = pytest_fixtures

        # Step 3: Extract test functions from the generated unit test block
        self._extract_test_case_from_test_cases(
            llm_prompt_executor,
            cls_setting.llm_extract_test_cases_prompt,
            generated_unit_test_code
        )


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


    def init_variables(self, source_code_file: str, cls_settings: cls_Settings) -> Tuple[int, str, str, str, cls_SourceCode]:
        """
        Initializes variables required for processing a source code file's unit test generation.

        Args:
            source_code_file (str): Path to the source code file to be processed.
            cls_settings (cls_Settings): Settings object containing configuration and paths.

        Returns:
            Tuple[int, str, str, str, cls_SourceCode]: A tuple containing:
                - passed_count (int): Counter for passed test cases, initialized to 0.
                - overall_error_msg (str): Aggregated error message string.
                - successful_import_stmt (str): Accumulated successful import statements.
                - success_unit_test (str): Accumulated successful unit test code.
                - cls_source_code (cls_SourceCode): Source code wrapper object initialized for processing.
        """
        passed_count = 0
        overall_error_msg = ""
        successful_import_stmt = ""
        success_unit_test = ""
        cls_source_code = cls_SourceCode(source_code_file, cls_settings)

        return passed_count, overall_error_msg, successful_import_stmt, success_unit_test, cls_source_code


    def derive_test_cases(
        self,
        source_dir: Path,
        cls_source_code: cls_SourceCode,
        cls_settings: cls_Settings
    ) -> int:
        """
        Derives unit test cases from the given source code using an LLM if applicable.

        If the source code qualifies for unit test generation, it uses the LLM to generate
        test cases, saves the generated code to the output directory, and updates the test
        case components. If not, it marks the file as skipped with a remark.

        Args:
            source_dir (Path): The source directory where the input code file is located.
            cls_source_code (cls_SourceCode): The parsed source code object.
            cls_settings (cls_Settings): Settings object containing configurations.

        Returns:
            int: The number of test cases derived (length of self.unit_test).
        """
        if self._should_generate_unit_test(cls_source_code.source_code):
            llm_prompt_executor = LLMPromptExecutor(cls_settings)

            # Generate unit test code using LLM
            generated_unit_test_code = self._generates_test_cases(
                cls_source_code,
                cls_settings,
                llm_prompt_executor
            )

            # Save generated test code to file
            savefile = SaveFile(source_dir, cls_source_code.source_code_file_path)
            savefile.save_file(Path(cls_settings.generated_tests_dir), generated_unit_test_code, "init_")

            # Build test components
            self._build_unit_test_code(
                generated_unit_test_code,
                cls_source_code,
                cls_settings,
                llm_prompt_executor
            )

            self.remarks = ""
        else:
            self.remarks = "Skipped (config/models/imports only)"
            logger.warning(
                f"Skipping unit test generation for {cls_source_code.source_code_file_path} "
                "as it only contains configuration, models, or pure imports."
            )

        return len(self.unit_test)
    