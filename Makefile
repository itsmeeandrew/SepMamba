## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.11.7
PYTHON_INTERPRETER = python3
ENV_NAME = speaker-separation
LOADED_PYTHON_VERSION := $(shell python3 --version 2>&1 | cut -d ' ' -f 2)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment (conda)
create_conda_environment:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

# Set up python interpreter environment (venv)
create_venv_environment:
	if [ $(LOADED_PYTHON_VERSION) != $(PYTHON_VERSION) ]; then \
		echo "Loaded python version is not $(PYTHON_VERSION). Please load the appropriate python version"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --no-cache-dir
	echo "Mamba can only be installed if nvcc is available. If you are using HPC, don't forget to load the cuda module"
	$(PYTHON_INTERPRETER) -m pip install -r requirements_mamba.txt --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install .



## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')