from pathlib import Path
# from TestPilot.logger import setup_logger
from TestPilot.common_helpers import setup_logger, cls_Settings
# from TestPilot.settings import cls_Settings

logger = setup_logger()
from pathlib import Path

class cls_SourceCode:
    def __init__(self, source_code_file_path:Path, settings:cls_Settings):
        self.source_code = ""
        self.source_code_file_path = source_code_file_path
        self.python_version = settings.python_version
        self.requirements_txt = settings.requirements_txt
        self.module_path=".".join(Path(self.source_code_file_path).with_suffix("").parts)

    def process_source_file(self) -> None:
        logger.info(f"process_source_file starts")
        # Read source code
        self.source_code = self.source_code_file_path.read_text(encoding="utf-8")
        logger.info(f"process_source_file completes")
        return None
