#!/bin/bash
clear
pre-commit uninstall
git add .
git commit -m "test: test"
pre-commit install
git commit --amend --no-edit
