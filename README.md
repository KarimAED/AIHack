# AIHack

## A Project submission by K. Alaa El-Din, L. Versini, S. Chen and A. Poon

This is our submission for approaching the 2021 Imperial College AI Hack Crop yield challenge.
All of the data figures presented in our report were generated using this code.

## Usage

To run this project, install dependencies from pyproject.toml, or manually:

Required:
	numpy@^1.19.3
	pandas@^1.2.2
	scipy@^1.6.1
	tensorflow@^2.4.1
	sklearn@^0.0

For some of the plotting seaborn and geopandas may also be required (gdal DLL must be installed under Windows).

### Main Pipeline
You can then proceed to follow the main CNN pipeline.

To generate all processed datasets again and some of the plots (optional, as datasets are included):

1. run EVI_data_reshaping.py and time_interpolation.py to generate 2 datasets aligned in time through interpolation
2. run interpolate_space.py to generate an input dataset that is aligned in time and space
3. run process_join.py to process the input and output data further, sort them into events and save them into a joint files

To perform the neural network analysis:
4. run fit_seq.py, model fitting will be passed to stdout; and loss, mean absolute error + labels vs predictions will be plotted


### Other analysis + Plots


Use plots_temperature.py for further insights into the data and plots!


## Help

For any questions, please contact us under kka4718@ic.ac.uk !
