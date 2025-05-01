# Standard library
import logging
import os
import shutil
import sys
import re
import ast
import subprocess
from tabulate import tabulate
import pandas as pd 
from openai import OpenAI, AzureOpenAI
from pathlib import Path
from typing import Dict, Any, List, NoReturn, Union, Tuple, Optional

from generate_unit_test_objects.utils import EnvVarsLoader, LLMPromptExecutor, SaveFile
from TestPilot.models.settings import Settings
from generate_unit_test_objects.unit_test import SourceCodeFile, UnitTestComponent

os.makedirs("logs", exist_ok=True)
# Create file handler
file_handler = logging.FileHandler("logs/output.log")  # Make sure "logs/" exists or change path
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Configure logging accordingly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers (especially default StreamHandler from basicConfig)
if logger.hasHandlers():
    logger.handlers.clear()

# Add the file handler
logger.addHandler(file_handler)

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

def _initialize_llm(env_vars):
    logger.info(f"_initialize_llm start")
    provider = env_vars.llm_provider
    client = _get_llm_client(env_vars)
    model_arg = _get_model_arguments(
        provider=provider,
        model_name=env_vars.model_name,
        azure_deployment_id=env_vars.azure_deployment_id
    )
    llm_prompt_executor = LLMPromptExecutor(
        provider=client,
        model_arg=model_arg,
        llm_temperature=float(env_vars.llm_temperature))

    logger.info(f"_initialize_llm complete")
    return llm_prompt_executor

def _get_llm_client(env_vars):
    provider = env_vars.llm_provider.lower()

    if provider == "azure":
        return AzureOpenAI(
            api_key=env_vars.azure_openai_key,
            api_version=env_vars.azure_api_version,
            azure_endpoint=env_vars.azure_openai_endpoint,
        )

    if provider == "openai":
        openai_key = env_vars.openai_api_key
        return OpenAI(api_key=openai_key)
    raise ValueError(f"Unsupported provider: '{provider}'. Expected 'openai' or 'azure'.")


def _get_model_arguments(provider: str, model_name: str = "", azure_deployment_id: str = "") -> str:
    provider = provider.lower()

    if provider == "azure":
        if not azure_deployment_id:
            raise ValueError("azure_deployment_id must be provided for Azure OpenAI")
        return azure_deployment_id

    if provider == "openai":
        if not model_name:
            raise ValueError("model_name must be provided for OpenAI")
        return model_name

    raise ValueError(f"Unsupported provider: '{provider}'.")


def _get_python_files(directory: str) -> List[Path]:
    """
    Recursively retrieves all Python (.py) files within the given directory.

    Args:
        directory (str): The root directory to search for Python files.

    Returns:
        List[Path]: A list of Path objects representing all found .py files.
    """
    return list(Path(directory).rglob("*.py"))


def extract_function_class_and_factory_assignments(code: str) -> List[str]:
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

def convert_relative_to_absolute_imports(code: str, file_path: str) -> str:
    logger.info(f"Update relative import start")
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
    logger.info(f"Update relative import complete")
    return pattern.sub(replacer, code)


def construct_module_import(function_names: List[str], source_code_path: str) -> str:
    """
    Generates a dynamic import statement based on the provided function names
    and source file path.

    Args:
        function_names (List[str]): Names of functions to import.
        source_code_path (str): Source file path (e.g., 'theory_evaluation/llm_utils.py').

    Returns:
        str: Formatted import statement (e.g., 'from theory_evaluation.llm_utils import func1, func2').
    """
    logger.info("construct_module_import starts")

    # Convert to dotted module path and remove .py extension
    module_path = os.path.splitext(source_code_path)[0].replace(os.sep, ".")

    names_str = ", ".join(function_names)
    import_line = f"from {module_path} import {names_str}"

    logger.info("construct_module_import completes")
    return import_line


def rewrite_module_references(source_code_path: str, source_code: str) -> str:
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


def save_transformed_file_path(
    source_dir: Path,
    target_dir: Path,
    original_path: Path,
    content: str,
    prefix: str = "test_",
    file_extension: Optional[str] = ".py"
) -> Path:
    relative_path = original_path.relative_to(source_dir)
    new_name = f"{prefix}{relative_path.stem}"
    transformed_path = target_dir / relative_path.parent / new_name

    if file_extension:
        transformed_path = transformed_path.with_suffix("." + file_extension.lstrip("."))

    transformed_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_path.write_text(content, encoding="utf-8")
    return transformed_path


def extract_test_functions(code: str) -> List[str]:
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


def run_single_test_file(temp_path: Path) -> Tuple[bool, str]:
    """
    Runs pytest on a file that contains a single test function.

    Args:
        temp_path (Path): Path to the test file.

    Returns:
        Tuple[bool, str]: (test_passed, test_output)
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    result = subprocess.run(
        ["pytest", str(temp_path), "--tb=short", "--quiet"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    passed = result.returncode == 0
    return passed, result.stdout.strip()

def extract_imported_symbols(test_code: str) -> List[Tuple[str, str]]:
    """
    Extracts imported symbols and their module paths from the test code.

    Returns:
        List of tuples: (symbol, module_path)
    """
    imports = []
    tree = ast.parse(test_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports.append((alias.name, node.module))
    return imports

   
def prepare_and_run_test_case(
    idx: int,
    retry_count: int,
    full_unit_test: str,
    temp_test_path: Path,
    source_code_path
) -> tuple[bool, str]:
    logger.info(f"prepare_and_run_test_case starts")
    
    output_str = f"\nTEST CASE {idx} Retry {retry_count} -- {source_code_path}\n---------------\n{full_unit_test}\n---------------\n\n"

    if os.path.exists(temp_test_path):
        os.remove(temp_test_path)
    
    os.makedirs(os.path.dirname(temp_test_path), exist_ok=True)
    full_unit_test=rewrite_module_references(source_code_path, full_unit_test)
    temp_test_path.write_text(f"# {source_code_path}\n{full_unit_test}", encoding="utf-8")
    passed, test_case_error = run_single_test_file(temp_test_path)

    output_str += f"TEST CASE {idx} Retry {retry_count} - Result - {'Passed' if passed == 1 else 'Failed'}\n\n"

    if not passed:
        output_str += f"Test Error -\n{test_case_error}\n\n"
    logger.info(f"prepare_and_run_test_case completes")
    return passed, test_case_error, output_str


def run_each_pytest_function_individually(
    idx,
    env_vars,
    unit_test_component:UnitTestComponent,
    savefile:SaveFile,
    llm_prompt_executor,
    source_code_file:SourceCodeFile
):
    logger.info(f"run_each_pytest_function_individually start")
    ret_val = ""
    failure_test_cases = ""
    retry_count = 0
    max_retries = int(env_vars.max_num_tries)
    unit_test_component.ensure_import_statements()

    full_unit_test = unit_test_component.full_unit_test
    passed, test_case_error, output_str = prepare_and_run_test_case(idx, 0, full_unit_test,Path(env_vars.temp_test_file), source_code_file.source_code_path)
    if not passed:
        failure_test_cases += output_str + "\n"
    logger.info(f"{output_str}")

    while retry_count <= max_retries and not passed:
        retry_count += 1
        llm_parameter = {
            "source_code": source_code_file.source_code,
            "test_case": full_unit_test,
            "test_case_error": test_case_error,
            "requirements_txt": source_code_file.requirements_txt,
        }
        full_unit_test = llm_prompt_executor.execute_llm_prompt(env_vars.llm_resolve_non_orm_prompt, llm_parameter)
        full_unit_test = update_patch_targets(full_unit_test, source_code_file.source_code_path)
        test_case = extract_test_case_from_test_cases(
            llm_prompt_executor, env_vars.llm_extract_test_cases_prompt, 
            full_unit_test,True)
        unit_test_component.update_unit_test(test_case)
        unit_test_component.ensure_import_statements()
        full_unit_test = unit_test_component.full_unit_test
        passed, test_case_error, output_str = prepare_and_run_test_case(idx, retry_count, full_unit_test,Path(env_vars.temp_test_file), source_code_file.source_code_path)
        if not passed:
            failure_test_cases += output_str + "\n"
        logger.info(f"{output_str}")
    if passed:
        ret_val = unit_test_component.unit_test
    else:
        ret_val = failure_test_cases
    
    logger.info(f"run_each_pytest_function_individually complete")
    return passed, ret_val

def prepare_import_statements(llm_prompt_executor, env_vars, source_code_file, generated_unit_test_code):
    logger.info(f"prepare_import_statements starts")
    llm_parameter = {"source_code": source_code_file.source_code}
    source_code_import_statements = llm_prompt_executor.execute_llm_prompt(
        env_vars.llm_extract_import_prompt, llm_parameter)
    source_code_import_statements = convert_relative_to_absolute_imports(source_code_import_statements, source_code_file.source_code_path)
    function_names = extract_function_class_and_factory_assignments(source_code_file.source_code)
    dynamic_imports = construct_module_import(function_names, source_code_file.source_code_path)
    source_code_import_statements += "\n" + dynamic_imports + "\n"

    llm_parameter = {"source_code": generated_unit_test_code}
    unit_test_import_statements = llm_prompt_executor.execute_llm_prompt(
        env_vars.llm_extract_import_prompt, llm_parameter)

    llm_parameter = {"source_code_import_statements": source_code_import_statements,
                     "unit_test_import_statements": unit_test_import_statements}
    import_statements = llm_prompt_executor.execute_llm_prompt(
        env_vars.llm_merge_imports_prompt, llm_parameter)
    logger.info(f"prepare_import_statements completes")
    return import_statements

def should_generate_unit_test(code: str) -> bool:
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
            if line.strip() and not re.match(r'^(from\s+\.\s+import\s+\w+|import\s+\w+)', line.strip())
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

def _generate_unit_tests_from_source_code(
        llm_prompt_executor,
        prompt,
        source_code_file
                ) -> Tuple[str, str]:
    logger.info(f"Generate Unit Test Case starts")

    llm_parameter = {"python_version": source_code_file.python_version, 
                     "requirements_txt": source_code_file.requirements_txt,
                     "file_content": source_code_file.source_code,
                     "module_path": source_code_file.module_path
                     }
    generated_unit_test_code = llm_prompt_executor.execute_llm_prompt(prompt, llm_parameter)

    logger.info(f"Generate Unit Test Case completes")
    return generated_unit_test_code

def extract_test_case_from_test_cases(
    llm_prompt_executor,
    prompt,
    generated_unit_test_code: str,
    output_one_case:Optional[bool] 
):
    llm_parameter = {"unit_test_code": generated_unit_test_code}
    all_test_cases_str = llm_prompt_executor.execute_llm_prompt(
        prompt,
        llm_parameter
    )
    all_test_cases_list = extract_test_functions(all_test_cases_str)
    if output_one_case:
        return all_test_cases_list[0]
    else:
        return all_test_cases_list


def update_patch_targets(generated_unit_test_code: str, source_code_path: str) -> str:
    """
    Replace unqualified module references (like 'llm_handler') in quoted strings
    (e.g. in @patch decorators) with fully qualified paths (e.g. 'theory_evaluation.llm_handler').

    Args:
        generated_unit_test_code (str): The generated test code.
        source_code_path (str): Path to the module, e.g. 'theory_evaluation/llm_handler.py'.

    Returns:
        str: The updated unit test code.
    """
    path = Path(source_code_path).with_suffix("")  # Remove .py
    full_module_path = ".".join(path.parts)
    module_name = path.stem

    # Match quoted strings that contain the bare module name, not already qualified
    pattern = rf"(['\"])({module_name})(?=\.[^'\"]*)(?<!{re.escape(full_module_path)})(?=\.[^'\"]*\1)"

    return re.sub(pattern, rf"\1{full_module_path}", generated_unit_test_code)


def _process_source_file(source_code_path, llm_prompt_executor, env_vars) -> None:
    logger.info(f"\nStart Processing file: {source_code_path}")
    
    # Extracts import 
    total_test_case=0
    passed_count = 0
    remarks=""
    unit_test_component = None
    source_code = source_code_path.read_text(encoding="utf-8")
    source_code_file = SourceCodeFile(Path(source_code_path), source_code, env_vars.python_version, 
                                        env_vars.requirements_txt)
    savefile = SaveFile(Path(env_vars.source_dir), source_code_file.source_code_path)
    if should_generate_unit_test(source_code_file.source_code):
        if env_vars.read_from_unit_test == "1":
            generated_unit_test_code = Path(env_vars.unit_test_file).read_text(encoding="utf-8")
        else:
            generated_unit_test_code  = _generate_unit_tests_from_source_code(
                llm_prompt_executor,
                prompt=env_vars.llm_generate_unit_tests_prompt,
                source_code_file=source_code_file
            )
        savefile.save_file(Path(env_vars.generated_tests_dir), generated_unit_test_code)
        
        generated_unit_test_code = update_patch_targets(generated_unit_test_code, source_code_file.source_code_path)
        
        import_statement = prepare_import_statements(llm_prompt_executor, env_vars, source_code_file, generated_unit_test_code)
        unit_test_component = UnitTestComponent(import_statement)
        llm_parameter={"unit_test_code": generated_unit_test_code}
        pytest_fixtures = llm_prompt_executor.execute_llm_prompt(env_vars.llm_extract_pytest_fixture_prompt, llm_parameter)
        if pytest_fixtures:
            unit_test_component.set_pytest_fixtures(pytest_fixtures)
            unit_test_component.append_import("\nimport pytest")
        
        generated_unit_test_code = convert_relative_to_absolute_imports(generated_unit_test_code, source_code_file.source_code_path)
        all_test_cases_list = extract_test_case_from_test_cases(
            llm_prompt_executor, env_vars.llm_extract_test_cases_prompt, 
            generated_unit_test_code, False)
        total_test_case = len(all_test_cases_list)
        logger.info(f"Number of test case to process - {total_test_case}")

        success_test_cases=""
        failure_test_cases=""
        for idx, test_case in enumerate(all_test_cases_list, start=1):
            unit_test_component.set_unit_test(test_case)
            passed, ret_val = \
                run_each_pytest_function_individually(idx, env_vars, unit_test_component, savefile, 
                                                      llm_prompt_executor, source_code_file)
            if passed:
                success_test_cases += ret_val + "\n\n"
                passed_count += 1
            else:
                failure_test_cases += ret_val + "\n\n"
        
        full_unit_tests = ""
        if unit_test_component.import_statement:
            full_unit_tests += unit_test_component.import_statement + "\n\n"
        if unit_test_component.pytest_fixtures:
            full_unit_tests += unit_test_component.pytest_fixtures + "\n\n"
        if success_test_cases:
            full_unit_tests += success_test_cases + "\n\n"
        llm_parameter={"full_unit_tests": full_unit_tests}
        full_unit_tests = llm_prompt_executor.execute_llm_prompt(env_vars.llm_format_test_code_prompt, llm_parameter)
        savefile.save_file(Path(env_vars.finalized_tests_dir), full_unit_tests)
        savefile.save_file(Path(env_vars.failed_tests_dir), failure_test_cases,file_extension=".log")
    else:
        remarks = "Skipped (config/models/imports only)"
        logger.warning(f"Skipping unit test generation for {source_code_path} as it only contains configuration, models, or pure imports.")

    logger.info(f"\nEnd Processing file: {source_code_path}\n")
    return passed_count, total_test_case, remarks

def _initialize_application():
    try:
        logger.info(f"_initialize_application start")
        env_vars = EnvVarsLoader()
        clean_test_environment(env_vars)

        llm_prompt_executor = _initialize_llm(env_vars)
        source_code_files = _get_python_files(env_vars.source_dir)

        logger.info(f"_initialize_application complete")
        return env_vars, llm_prompt_executor, source_code_files
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

def pre_processing() -> Tuple['Settings', List[Path]]:
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
    logger.info(f"pre_processing start")
    # Read Settings
    settings = Settings()
    # Read Housekeep Prcocessing Folders
    clean_test_environment(settings)
    # Read directory
    source_code_files = _get_python_files(settings.source_dir)
    logger.info(f"pre_processing end")
    return settings, source_code_files

def main() -> NoReturn:
    settings, source_code_files = pre_processing() 

    test_stats = []
    for source_code_file in source_code_files:
        logger.info(f"Hello World - XXX")

    #     passed_count, total_test_case, remarks = _process_source_file(source_code_file, llm_prompt_executor, env_vars)
    #     test_stats.append({
    #         "filename": source_code_file,
    #         "total_test_cases_passed": passed_count,
    #         "total_test_cases": total_test_case,
    #         "percentage_passed (%)": (passed_count / total_test_case * 100) if total_test_case > 0 else 0.0,
    #         "remarks": remarks
    #     })

    # test_stats_df = pd.DataFrame(test_stats)
    # test_stats_df.index = test_stats_df.index + 1
    
    # logger.info("\n" + tabulate(test_stats_df, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
    finally:
        sys.exit(0)        
