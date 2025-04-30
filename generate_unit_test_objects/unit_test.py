from typing import Dict
import re

IMPORT_MAP = {
    "pytest": "import pytest",
    "patch": "from unittest.mock import patch",
    "mock_open": "from unittest.mock import mock_open",
    "mock": "from unittest import mock",
    "AsyncMock": "from unittest.mock import AsyncMock",
    "MagicMock": "from unittest.mock import MagicMock"
}

class SourceCodeFile:
    def __init__(self, source_code_path, source_code, python_version, requirements_txt):
        self.source_code = source_code
        self.source_code_path = source_code_path
        self.python_version = python_version
        self.requirements_txt = requirements_txt

class UnitTestComponent:
    def __init__(self, import_statement):
        self.import_statement = import_statement
        self.pytest_fixtures = ""
        self.unit_test = ""
        self.full_unit_test = ""

    def set_pytest_fixtures(self, pytest_fixtures):
        self.pytest_fixtures = pytest_fixtures

    def set_unit_test(self, unit_test):
        self.unit_test = unit_test
        self.full_unit_test = self.import_statement + "\n\n" + self.pytest_fixtures + "\n\n" + self.unit_test

    def ensure_import_statements(self) -> None:
        for keyword, import_line in IMPORT_MAP.items():
            if (keyword in self.unit_test or keyword in self.pytest_fixtures) and import_line not in self.import_statement:
                self.import_statement += f"\n{import_line}"
        self.full_unit_test = self.import_statement + "\n\n" + self.pytest_fixtures + "\n\n" + self.unit_test

    def update_unit_test(self, unit_test) -> None:
        self.unit_test = unit_test
        self.full_unit_test = self.import_statement + "\n\n" + self.pytest_fixtures + "\n\n" + self.unit_test

    def append_import(self, import_statement) -> None:
        self.import_statement += f"\n{import_statement}"
        self.full_unit_test = self.import_statement + "\n\n" + self.pytest_fixtures + "\n\n" + self.unit_test

        