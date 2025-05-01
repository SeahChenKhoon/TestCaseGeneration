from TestPilot.utils.logger import setup_logger 
from TestPilot.models.source_file import SourceCodeFile
logger = setup_logger()
from pathlib import Path

class SourceCodeBiz:
    def process_source_file(source_code_path, env_vars) -> None:
        logger.info(f"process_source_file starts")
        # Read source code
        source_code = source_code_path.read_text(encoding="utf-8")
        # Create Source code object
        source_code_file = SourceCodeFile(Path(source_code_path), source_code, env_vars.python_version, 
                                            env_vars.requirements_txt)
        logger.info(f"process_source_file completes")
        return source_code_file
