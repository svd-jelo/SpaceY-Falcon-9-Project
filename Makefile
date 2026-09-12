#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = spacey_falcon_9_project
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Create Dataset
.PHONY: dataset
dataset:
	$(PYTHON_INTERPRETER) spacey_falcon_9_project/dataset.py

## Train model over dataset
.PHONY: train_model
train_model:
	$(PYTHON_INTERPRETER) spacey_falcon_9_project/modeling/train.py

## Make batch predictions
.PHONY: batch_predict
batch_predict:
ifdef MODEL_NAME
	$(PYTHON_INTERPRETER) spacey_falcon_9_project/modeling/predict.py \
		--model_name $(MODEL_NAME) \
		$(if $(INPUT_PATH),--input_path $(INPUT_PATH)) \
		$(if $(OUTPUT_PATH),--output_path $(OUTPUT_PATH)) \
		$(if $(MODEL_DIR),--model_dir $(MODEL_DIR))
else
	@echo "Error: MODEL_NAME is required. Usage: make batch_predict MODEL_NAME=<filename>"
	@exit 1
endif

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
