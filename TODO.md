# Functional

- [ ] (Regarding Supplements) Adapt and test code for residualization (regressed_features.ipynb) and rolling correlation analysis (continuous_perf.ipynb).
- [ ] Add README.md
- [ ] Test minimal environment, and installation instructions: 


# Organizational

- [ ] Unsure how to provide or link to raw data, since it is several GB per folder. Without it the code would not run anyways. Provide only the pre-processed data? Only the resulting model files should be equivalent to 1-2 GB total.

# Improvements

- [ ] Change all instances of "DN" in filenames, tables and variables, since only one neuron is DN.
- [ ] Clean names to DopaMeander, in general and in particular when loading tables with load_into_pandas.
- [x] Clean tools script.
- [ ] Comment steps within code and functions.
- [ ] Divide code into modules for better readibility.
- [ ] Remove "avoids specific cases in which the specific variable in the specific experiment is constant" in analysis.ipynb if confirmed to not be a problem anymore in the current dataset.
