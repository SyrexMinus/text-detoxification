Makar Shevchenko | m.shevchenko@innopolis.university | B20-DS-01

# text-detoxification | PMLDL course | Assignment 1
Transforming toxic style text into neutral style text preserving meaning.

In this work I analysed the ParaNMT-detox corpus and applied it for training of various detoxification
models starting from simple dictionary-based methods ending with a recurrent and attention-based sequence network.

## Navigation in the Repo

The task that I implemented is described in `references/Solution_Process.md`.

The description of tried approaches are in `reports/Solution_Process.md`, the report on the final recurrent sequence 
network solution is `reports/Final_Solution.md`.

The notebooks on data exploration & preprocessing and on model training & evaluation are in `notebooks`.

The list of resources used is listed in `references/references.bib`.

The weights of the trained models are in `models`.

The `data` directory is used for storage of data for scripts. `external` subdirectory contain data from third party
sources, `inernal` - intermediate data that has been transformed, and `raw` - the original, immutable data.

## Usage

Before executing any script make sure that your Python is of version 3.10.3 or above. All the commands should be 
executed from the repo root. The commands are tested on MacOS 13.5.2.

1. `pip install -r requirements.txt`

### Data Loading & Preprocessing

1. `python src/data/make_dataset.py`

The script will store raw dataset in `data/raw` and train and validation files in `data/internal`

### Model Training

0. Go through Data Loading & Preprocessing

1. `python src/models/train_model.py`

The script will train the final recurrent sequence network solution described is `reports/Final_Solution.md` and
store the trained weights in `models`.

### Predicting

0. Go through Data Loading & Preprocessing

1. `python src/models/predict_model.py "my dirty toxic sentence"`
2. Wait for the last line of the output - it will be detoxified sentence
