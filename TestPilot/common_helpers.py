import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import logging

def setup_logger(name: str = "TestPilot") -> logging.Logger:
    """
    Sets up and returns a logger instance with console and file handlers.

    Args:
        name (str): Name of the logger. Defaults to 'TestPilot'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "output.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent duplicate handlers if imported multiple times
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

class SaveFile:
    def __init__(self, source_dir, original_path):
        self.source_dir = source_dir
        self.original_path = original_path

    def save_file(self, target_dir, file_content, prefix = "test_", file_extension: Optional[str] = ".py"):
        relative_path = self.original_path.relative_to(self.source_dir)
        new_name = f"{prefix}{relative_path.stem}"
        transformed_path = Path(target_dir) / relative_path.parent / new_name

        if file_extension:
            transformed_path = transformed_path.with_suffix("." + file_extension.lstrip("."))

        transformed_path.parent.mkdir(parents=True, exist_ok=True)
        transformed_path.write_text(file_content, encoding="utf-8")
        return "transformed_path"                        

class cls_Settings:
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
        self.max_num_tries = int(os.getenv("MAX_NUM_TRIES"))
        self.should_generate_tests = bool(int(os.getenv("SHOULD_GENERATE_TESTS", "1")))
        self.unit_test_file = os.getenv("UNIT_TEST_FILE")
        self.source_file = os.getenv("SOURCE_FILE")
        self.required_imports = os.getenv("REQUIRED_IMPORTS")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE"))
        self.llm_generate_unit_tests_prompt = os.getenv("LLM_GENERATE_UNIT_TESTS_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_extract_import_prompt = os.getenv("LLM_EXTRACT_IMPORT_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_resolve_prompt = os.getenv("LLM_RESOLVE_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_extract_pytest_fixture_prompt = os.getenv("LLM_EXTRACT_PYTEST_FIXTURE_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_extract_test_cases_prompt = os.getenv("LLM_EXTRACT_TEST_CASES_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_merge_imports_prompt = os.getenv("LLM_MERGE_IMPORTS_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_organize_imports_prompt = os.getenv("LLM_ORGANIZE_IMPORTS_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.llm_merge_unittest_existing_imports_prompt = \
            os.getenv("LLM_MERGE_UNITTEST_EXISTING_IMPORTS_PROMPT") + os.getenv("LLM_TRAILER")
        self.llm_merge_pytest_fixtures_prompt = os.getenv("LLM_MERGE_PYTEST_FIXTURES_PROMPT") + \
            os.getenv("LLM_TRAILER")
        self.requirements_txt = Path("./requirements.txt").read_text(encoding="utf-8")    


class LLMPromptExecutor:
    def __init__(
        self, cls_setting:cls_Settings
    ):
        provider = cls_setting.llm_provider
        client = self._get_llm_client(cls_setting)
        model_arg = self._get_model_arguments(
            provider=provider,
            model_name=cls_setting.model_name,
            azure_deployment_id=cls_setting.azure_deployment_id
        )
        
        self.provider=client
        self.model_arg=model_arg
        self.llm_temperature=cls_setting.llm_temperature


    def _get_llm_client(self, env_vars):
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

    def _get_model_arguments(self, provider: str, model_name: str = "", azure_deployment_id: str = "") -> str:
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


    def _get_chat_completion(self, provider: Any, model: str, prompt: str, llm_temperature: float = 0.2) -> Any:
        """
        Sends a prompt to the chat model and returns the response.

        Args:
            provider (Any): The provider instance with a chat.completions.create method.
            model (str): The model name to use.
            prompt (str): The user prompt to send.
            llm_temperature (float, optional): Sampling llm_temperature. Defaults to 0.2.

        Returns:
            Any: The response object from the provider.
        """
        return provider.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_temperature,
        )

    def _strip_markdown_fences(self, text: str) -> str:
        """
        Removes all Markdown-style triple backtick fences from LLM output.
        Logs a warning if any stripping was performed.

        Args:
            text (str): The raw LLM output string.

        Returns:
            str: The cleaned string without Markdown-style code fences.
        """
        lines = text.strip().splitlines()
        cleaned_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                continue  # Skip the fence line
            cleaned_lines.append(line)


        return "\n".join(cleaned_lines)

    def execute_llm_prompt( 
        self,
        llm_import_prompt: str,
        llm_parameter: dict,
    ) -> str:
        formatted_prompt = llm_import_prompt.format(**llm_parameter)
        response = self._get_chat_completion(self.provider, self.model_arg, formatted_prompt, self.llm_temperature)
        return self._strip_markdown_fences(response.choices[0].message.content.strip())
    