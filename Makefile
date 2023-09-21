.PHONY: *

PYTHON_EXEC := python
DROPBOX_DATASET := .dropbox_dataset

CLEARML_PROJECT_NAME := image_classification
CLEARML_DATASET_NAME := image_classification_dataset


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python
	poetry shell


migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(DROPBOX_DATASET) || true
	mkdir $(DROPBOX_DATASET)
	wget "https://www.dropbox.com/scl/fi/3x9eksnjcjvsy1ey35wd8/Classification_data.zip?rlkey=j9qeriu1g2dwhc2iwb9a8b6ql&dl=0" -O $(DROPBOX_DATASET)/dataset.zip
	unzip -q $(DROPBOX_DATASET)/dataset.zip -d $(DROPBOX_DATASET)
	rm $(DROPBOX_DATASET)/dataset.zip
	find $(DROPBOX_DATASET) -type f -name '.DS_Store' -delete
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(DROPBOX_DATASET)
	clearml-data close --verbose
	rm -R $(DROPBOX_DATASET)


run_training:
	poetry run $(PYTHON_EXEC) -m src.train
