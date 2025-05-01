from pathlib import Path
from dotenv import load_dotenv
import os

class Settings:
    def __init__(self):
        load_dotenv(override=True)
        self.llm_provider = os.getenv("LLM_PROVIDER")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.source_dir_str = os.getenv("SOURCE_DIR")
        self.generated_tests_dir = Path(os.getenv("GENERATED_TESTS_DIR"))
        self.finalized_tests_dir = Path(os.getenv("FINALIZED_TESTS_DIR"))
        self.temp_test_file = Path(os.getenv("TEMP_TEST_FILE"))
        self.log_file = Path(os.getenv("LOG_FILE"))
        self.failed_tests_dir = Path(os.getenv("FAILED_TESTS_DIR"))
        self.model_name = os.getenv("MODEL_NAME")
        self.python_version = os.getenv("PYTHON_VERSION")
        self.max_num_tries = os.getenv("MAX_NUM_TRIES")
        self.read_from_unit_test = os.getenv("READ_FROM_UNIT_TEST")
        self.unit_test_file = os.getenv("UNIT_TEST_FILE")
        self.source_file = os.getenv("SOURCE_FILE")
        self.required_imports = os.getenv("REQUIRED_IMPORTS")
        self.llm_temperature = os.getenv("LLM_TEMPERATURE")
        self.llm_classify_orm_prompt = os.getenv("LLM_CLASSIFY_ORM_PROMPT")
        self.llm_format_test_code_prompt = os.getenv("LLM_FORMAT_TEST_CODE_PROMPT")
        self.llm_test_orm_prompt = os.getenv("LLM_TEST_ORM_PROMPT")
        self.llm_generate_unit_tests_prompt = os.getenv("LLM_GENERATE_UNIT_TESTS_PROMPT")
        self.llm_extract_import_prompt = os.getenv("LLM_EXTRACT_IMPORT_PROMPT")
        self.llm_unique_import_prompt = os.getenv("LLM_UNIQUE_IMPORT_PROMPT")
        self.llm_resolve_orm_prompt = os.getenv("LLM_RESOLVE_ORM_PROMPT")
        self.llm_resolve_non_orm_prompt = os.getenv("LLM_RESOLVE_NON_ORM_PROMPT")
        self.llm_extract_pytest_fixture_prompt = os.getenv("LLM_EXTRACT_PYTEST_FIXTURE_PROMPT")
        self.llm_extract_test_cases_prompt = os.getenv("LLM_EXTRACT_TEST_CASES_PROMPT")
        self.llm_cleanup_prompt = os.getenv("LLM_CLEANUP_PROMPT")
        self.llm_merge_imports_prompt = os.getenv("LLM_MERGE_IMPORTS_PROMPT")
        self.requirements_txt = Path("./requirements.txt").read_text(encoding="utf-8")    

        