# GLM models

## Installation of python environment

```
mamba create -n mdndm_env "python==3.12.8" "scipy==1.15.0" "scikit-learn==1.6.0" tqdm numpy pandas ipykernel openpyxl matplotlib seaborn -c conda-forge
mamba activate mdndm_env
pip install glm_utils
```

## Raw data

Acquire raw data and save in a folder with the structure "./neuron" per neuron group, where each experiment is saved as "./neuron/experiment_unique_datename.csv" and must also contain a table "./neuron/neuron.xlsx" with all corresponding metadata.

## Figures

In relation to models in Figure 6.