from pathlib import Path

class SourceCodeFile:
    def __init__(self, source_code_path, source_code, python_version, requirements_txt):
        self.source_code = source_code
        self.source_code_path = source_code_path
        self.python_version = python_version
        self.requirements_txt = requirements_txt
        self.module_path=".".join(Path(self.source_code_path).with_suffix("").parts)


