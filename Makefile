PROJECT_NAME = ElementsOfDataScienceBook
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

.PHONY: compile_book

compile_book:
	pdflatex main.tex && evince main.pdf

create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

delete_environment:
	conda env remove --name $(PROJECT_NAME)

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 code
	black --check --config pyproject.toml code

format:
	black --config pyproject.toml code

tests:
	pytest --nbmake [01]*.ipynb
