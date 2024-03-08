# DEMOS

This repository contains code that runs simulations for the DEMOS model suite, as fully integrated with Urbansim models. The current DEMOS model suite is implemented for the San Francisco Bay Area and the Austin Area. Several demographic models are implemented in the current framework, in addition to several Urbansim models. Detailed documentation of the models will be added to this repository. Currently, the following demographic models are implemented:

- Aging Model: simulates the age evolution of the synthetic population
- Birth Model: simulates new births in the population in each year
- Mortality Model: simulates new mortalities in the population in each year
- Education Model: simulates dropouts out of the student population in each year
- Single to X Model: simulates changes in partnership status from Single to either cohabitating with a partner or getting married
- Cohabitation to X Model: simulates changes in partnership status from cohabitating with a partner to getting married or getting divorced
- Divorce Model: simulates new divorces in the synthetic population each year
- Kids leaving Household: simulates kids leaving their current households in each year
- In-Migration/Out-migration models: simulate households/individuals migrating into or out of the area each year
- Laborforce participation models: simulate labor force entry/exit of eligible workers each year
- Income growth models: simulates income growth for the population in each year

This repository is currently in active development and new updates will be continuously added. For any issues with the current codebase or to suggest improvements, please open an issue or create a separate branch to add your feature and submit pull request for review.


## I. Setup Guide

This repository contains only code and configuration/setup files necessary 


1. Clone this repository into your local machine

2. Create Environment


3. Run the local setup

```
python3 setup.py develop
```

4. Install dependencies

The current dependencies installation is based on Unix based systems. Installation instructions for Windows systems will be added later.

It is also preferred to create a new environment to isolate the environment for this software from all other installed environments. This is to avoid creating any conflicts with already installed libraries or software. The recommended way to install environments is through Anaconda. Follow this guide to create a new environment and install the dependendecies in it.

```
apt-get update

apt-get install -y gcc libhdf5-serial-dev
```

Install Anaconda by following these instructions for [Linux](https://docs.anaconda.com/anaconda/install/linux/) or[MacOS](https://docs.anaconda.com/anaconda/install/mac-os/).

Create a conda environment
```
conda create -n DEMOS python=3.8
```
Activate the created conda environment

```
conda activate DEMOS
```
Install the project requirements.
```
pip install -r requirements.txt
```
If you run into any issues in setting up your environment, please submit an issue in this repository.

5. Download Relevant Data

Before running the simulation framework, create a data folder within the main project directory by typing the following in your terminal:

```
cd DEMOS

mkdir data
```

Access the data directory and download the relevant input data using the following:

- SF Bay Area
```
wget https://bayarea-urbansim.s3.us-east-2.amazonaws.com/custom_mpo_06197001_model_data.h5
wget -O bay_area_skims.csv.gz https://beam-outputs.s3.amazonaws.com/output/sfbay/sfbay-smartbaseline-WALK-TAZ-activitySimSkims-allTimePeriods-r5%2Bgh__2021-02-24_21-55-42_gnh/ITERS/it.0/0.activitySimODSkims.UrbanSim.TAZ.Full.csv.gz
```

6. SKIMS preprocessing

Before running any simulations, it's necessary to preprocess the skims files downloaded in Step 5 for your relevant region. To process the skims file, first activate your environment using the following command:
```
conda activate DEMOS_ENV
```

Access the main DEMOS project folder and running the skims processing file using the following command:
```
cd demos_urbansim
python process_skims.py
```

7. Running Simulation

Before running the simulation, navigate to the project directory and activate the created conda environment:

```
conda activate DEMOS
```

The general command for running the DEMOS simulation is the following:
```
python simulate.py -- -c -cf custom -l -sg -r <region ID> -i <input_year> -y <forecast year> -f <iter year frequency> -ss <skims source> -rm
```

The following are the arguments used in the above command:
```
-r REGION_CODE, --region_code REGION_CODE
                        region fips code
  -y YEAR, --year YEAR  forecast year to simulate to
  -c, --calibrated      whether to run with calibrated coefficients
  -cf CALIBRATED_FOLDER, --calibrated_folder CALIBRATED_FOLDER
                        name of the calibration folder to read configs from
  -sg, --segmented      run with segmented LCMs
  -l, --all_local       no cloud access whatsoever
  -i INPUT_YEAR, --input_year INPUT_YEAR
                        input data (base) year
  -f FREQ_INTERVAL, --freq_interval FREQ_INTERVAL
                        intra-simulation frequency interval
  -o OUTPUT_FNAME, --output_fname OUTPUT_FNAME
                        output file name
  -ss SKIM_SOURCE, --skim_source SKIM_SOURCE
                        skims format, e.g. "beam", "polaris"
  -rm, --random_matching Random matching in the Single to X model to 
                         reduce computational time due to 
                         the matchmaking process
```

This command will direct the software to the correct input data and configuration files and run the simulation based on the specified parameters. After the simulation is completed, the software will save output data and summary statistics in the output folder. The analyst can then use these saved objects to run any additional analyses necessary.

8. Simulation results
The DEMOS simulation will produce the following sets of data and results:
  - A synthetic population file containing the evolved synthetic population throughout the simulation years in `h5` format.

The file can be used to generate summary statistics on population characteristics for each of the simulated years.

## II. Project Structure

The main folder of this repository contains several python scripts that contain the different steps necessary to import, process, and run the DEMOS framework. The following is a description of the different folder and scripts used to run the DEMOS simulation

### The `configs\` directory
The `configs\` directory contains the different `.yaml` configuration files to run each of the DEMOS and urbansim models.
The configuration files follow the format based on UrbanSim Templates (https://github.com/UDST/urbansim_templates).
Each configuration file (see here for example) contains a model specification, the input and output tables, along with filtering conditions for input and output subpopulations (e.g.: when models are only applicable to subpopulations, such as the laborforce participation model)
The DEMOS models specification files are as follows:

1. `demos_birth.yaml`
2. `demos_mortality.yaml`
3. `demos_single_to_x.yaml`
4. `demos_married_to_x.yaml`


### The `data\` directory:
This directory contains the different data files needed to run the DEMOS model. Specifically, it includes the following files:

1. The input synthetic population for the region, 
2. Calibration data including observed outcomes between two specific years, with files in this nomenclature: `calibration_xxx_2010_2050.csv`
3. Post-processing mapping matrix to update household roles when applicable `.csv`
4. Income growth rates `.csv`

### `variables.py`
The file variables.py contains Python code defining and creating various types of variables necessary for use within the DEMOS models.
Each variable is defined as a function and decorated with an ```orca``` column dectorator, associated with agents such as persons and households.
For example, the following defines a dummy variable indicating whether each person is a female:

```
@orca.column('persons')
def gender2(persons):
    p = persons.to_frame(columns=['sex'])
    return p.eq(2).astype(int)
```
### `datasources.py`:
This script loads all the necessary data for the simulation, including the input data, models, and calibration data.

### `models.py`:
   This script defines all the models used in DEMOS, and listed in the beginning of this document. Each model is defined as an `orca` step. In addition to the models, the script contains pre-processing and post-processing functions for the models. Users can use this script to customize which models to run, and in what order. Users can add any new models here. An example of a model is as follows:

```
@orca.step("education_model")
def education_model(persons, year):
```

### `simulate.py`:
This script defines the simulation pipeline and the simulation parameters for DEMOS.

### `outputs\` directory:
This directory contains files with aggregate results produced by the simulation. Simulation results for each region are stored in their respective subdirectories.
