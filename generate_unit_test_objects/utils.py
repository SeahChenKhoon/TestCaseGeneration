import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

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

class EnvVarsLoader:
    def __init__(self):
        load_dotenv(override=True)
        self.llm_provider = os.getenv("LLM_PROVIDER")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.source_dir = os.getenv("SOURCE_DIR")
        self.generated_tests_dir = os.getenv("GENERATED_TESTS_DIR")
        self.finalized_tests_dir = os.getenv("FINALIZED_TESTS_DIR")
        self.temp_test_file = os.getenv("TEMP_TEST_FILE")
        self.log_file = os.getenv("LOG_FILE")
        self.failed_tests_dir = os.getenv("FAILED_TESTS_DIR")
        self.model_name = os.getenv("MODEL_NAME")
        self.python_version = os.getenv("PYTHON_VERSION")
        self.max_num_tries = os.getenv("MAX_NUM_TRIES")
        self.read_from_unit_test = os.getenv("READ_FROM_UNIT_TEST")
        self.unit_test_file = os.getenv("UNIT_TEST_FILE")
        self.source_file = os.getenv("SOURCE_FILE")
        self.required_imports = os.getenv("REQUIRED_IMPORTS")
        self.llm_temperature = os.getenv("LLM_TEMPERATURE")
        self.llm_classify_orm_prompt = os.getenv("LLM_CLASSIFY_ORM_PROMPT")
        self.llm_test_orm_prompt = os.getenv("LLM_TEST_ORM_PROMPT")
        self.llm_generate_unit_tests_prompt = os.getenv("LLM_GENERATE_UNIT_TESTS_PROMPT")
        self.llm_extract_import_prompt = os.getenv("LLM_EXTRACT_IMPORT_PROMPT")
        self.llm_unique_import_prompt = os.getenv("LLM_UNIQUE_IMPORT_PROMPT")
        self.llm_resolve_orm_prompt = os.getenv("LLM_RESOLVE_ORM_PROMPT")
        self.llm_resolve_non_orm_prompt = os.getenv("LLM_RESOLVE_NON_ORM_PROMPT")
        self.llm_extract_pytest_fixture_prompt = os.getenv("LLM_EXTRACT_PYTEST_FIXTURE_PROMPT")
        self.llm_extract_test_cases_prompt = os.getenv("LLM_EXTRACT_TEST_CASES_PROMPT")
        self.llm_cleanup_prompt = os.getenv("LLM_CLEANUP_PROMPT")
        self.requirements_txt = Path("./requirements.txt").read_text(encoding="utf-8")

class LLMPromptExecutor:
    def __init__(
        self,
        provider: str,
        model_arg: str,
        llm_temperature: float
    ):
        self.provider = provider
        self.model_arg = model_arg
        self.llm_temperature = llm_temperature


    def get_chat_completion(self, provider: Any, model: str, prompt: str, llm_temperature: float = 0.2) -> Any:
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

    def strip_markdown_fences(self, text: str) -> str:
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
        response = self.get_chat_completion(self.provider, self.model_arg, formatted_prompt, self.llm_temperature)
        return self.strip_markdown_fences(response.choices[0].message.content.strip())
    
    
