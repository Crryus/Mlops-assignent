.PHONY: tests docs
# This file is to auto run the codes

dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..." 
	poetry shell
# activate virtual env


# initialize_git

tests:
	pytest

#you can configure a alert here

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --force
	@echo View API documentation... 
	pdoc src --http localhost:8080	