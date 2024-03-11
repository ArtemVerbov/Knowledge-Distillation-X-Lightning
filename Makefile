.PHONY: *

DROPBOX_DATASET := .dropbox_dataset

CLEARML_PROJECT_NAME := image_classification
CLEARML_DATASET_NAME := image_classification_dataset

WEIGHTS_FOLDER := teacher_weights

migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(DROPBOX_DATASET) || true
	mkdir $(DROPBOX_DATASET)
	wget "https://www.dropbox.com/scl/fi/nrn0y41dsfwqsrssav2eo/Classification_data.zip?rlkey=oieytodt749yzyippc6384tge&dl=0" -O $(DROPBOX_DATASET)/dataset.zip
	unzip -q $(DROPBOX_DATASET)/dataset.zip -d $(DROPBOX_DATASET)
	rm $(DROPBOX_DATASET)/dataset.zip
	find $(DROPBOX_DATASET) -type f -name '.DS_Store' -delete
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	clearml-data add --files $(DROPBOX_DATASET)
	clearml-data close --verbose
	rm -R $(DROPBOX_DATASET)

download_weights:
	# Download pretrain weights for teacher model.
	rm -R $(WEIGHTS_FOLDER) || true
	mkdir $(WEIGHTS_FOLDER)
	wget "https://drive.usercontent.google.com/download?id=1VBOjPS0pn3IulzWSixCxJZE7mXWxxeJh&export=download&confirm=yes" -O $(WEIGHTS_FOLDER)/teacher.zip
	unzip -q $(WEIGHTS_FOLDER)/teacher.zip -d $(WEIGHTS_FOLDER)
	rm $(WEIGHTS_FOLDER)/teacher.zip

