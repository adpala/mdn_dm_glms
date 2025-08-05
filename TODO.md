# Functional

- [ ] (Supplements) Adapt and test code for residualization (regressed_features.ipynb), it will require a new analysis notebook and saving of results.

# Organizational

- [ ] Unsure how to provide or link to raw data, since it is several GB per folder. Without it the code would not run anyways. Provide only the pre-processed data? Only the resulting model files should be equivalent to 1-2 GB total.
- [ ] Reduce amount of data saved per model, to the minimum required.
- [ ] Save intermediate (downsampled) data to facilitate sharing it.

# Improvements

- [ ] Change all instances of "DN" in filenames, tables and variables, since only one neuron is DN.
- [ ] Clean names to DopaMeander, in general and in particular when loading tables with load_into_pandas.
- [ ] Comment steps within code and functions.
- [ ] Divide code into modules for better readibility.