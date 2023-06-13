# talos-rl

[![Tests](https://github.com/gepetto/talos-rl/actions/workflows/test.yml/badge.svg)](https://github.com/gepetto/talos-rl/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gepetto/talos-rl/main.svg)](https://results.pre-commit.ci/latest/github/gepetto/talos-rl/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

## Local + pip

### Install

Please use a `venv` or similar, with an up-to-date `pip`, and:

```
pip install .
```

### Run

```
python -m gym_talos -config config/config_RL.yaml
```

## Local + poetry

### Install

```
poetry install
```

### Run

```
poetry run -m gym_talos -config config/config_RL.yaml
```

## Local + conda

(here we use micromamba, but you can probably use your prefered conda flavor instead)

### Install

```
micromamba env create -f environment.yml
```

### Run

```
micromamba run -n talos-rl python -m gym_talos -config config/config_RL.yaml
```

## Apptainer + mamba

### Build

```
apptainer build rl_mamba.sif apptainer/rl_mamba.def
```

### Run

```
apptainer run rl_mamba.sif
```

## Apptainer + pip

Currently not available

## Apptainer + poetry

### Build

```
apptainer build rl.sif apptainer/rl.def
```

### Run

```
apptainer run rl.sif
```
