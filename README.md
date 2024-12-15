# continuity-contrastive-ecg

This codebase accompanies "Continuity Contrastive Representations of ECG for Heart Block Detection from Only Lead-I".

> **Note:**
> This repository is still in progress and will be finalized by December 31, 2024.

## Installation

To install this repository, from the project root, run:

```bash
pip install -e .
```

## Usage

### Command Line Interface

To train a model from scratch, run:

```bash
cc-ecg-train
```

To linear probe a pretrained model, run:

```bash
cc-ecg-lp
```

### Updating the configuration files

Some configurations must be defined by the user before the model can run. Primarily, this includes the dataset and label paths. These can be found in `/src/continuity_contrastive_ecg/configs/data/*.yaml` files or can be updated in the command line.
For example:

```bash
cc-ecg-train \
    data.label_dir=/path/to/labels/ \
    data.data_dir=/path/to/data/ \
```

This codebase accompanies "Continuity Contrastive Representations of ECG for Heart Block Detection from Only Lead-I".

> **Note:**
> This repository is still in progress and will be finalized by December 31, 2024.

## Installation

To install this repository, from the project root, run:

```bash
pip install -e .
```

## Usage

### Command Line Interface

To train a model from scratch, run:

```bash
cc-ecg-train
```

To linear probe a pretrained model, run:

```bash
cc-ecg-lp
```

### Updating the configuration files

Some configurations must be defined by the user before the model can run. Primarily, this includes the dataset and label paths. These can be found in `/src/continuity_contrastive_ecg/configs/data/*.yaml` files or can be updated in the command line.
For example:

```bash
cc-ecg-train \
    data.label_dir=/path/to/labels/ \
    data.data_dir=/path/to/data/ \
```
