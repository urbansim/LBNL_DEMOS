# DEMOS: Demographic Microsimulator


This repository contains code that runs simulations for the DEMOS model suite, as fully integrated with Urbansim models. The current DEMOS model suite is implemented for the San Francisco Bay Area and the Austin Area. Several demographic models are implemented along with  Urbansim models. Please refer to the following technical memorandum for more details on the implemented DEMOS models: [DEMOS Technical Memo](https://app.box.com/s/wcyo5jqcm4xzljtpyx72crml4uad8gey)

Current models implemented within DEMOS: 

- Aging Model: Increment the age of each agent in the simulation year 
- Birth Model: Predicts new births in households in each year
- Mortality Model: Predicts new mortalities in the population in each year
- Education Model: Predicts students who stop attending schools in each year
- Single to X Model: Predicts changes in partnership status from Single to either cohabitating with a partner or getting married
- Cohabitation to X Model: Predicts changes in partnership status from cohabitating with a partner to getting married or getting breakups
- Divorce Model: Predicts new divorces/separation in the synthetic population each year
- Kids leaving Household: Predicts which kids leave their parental households in each year
- In-Migration/Out-migration models: Predicts households/individuals migrating into or out of the area each year
- Laborforce participation models: Predicts employment status (i.e., employed or unemployed)of workers each year
- Income growth models: simulates income growth for the population in each year


# I. Project Structure
## models.py


The `models.py` file is the central script that contains all the modeling steps for the different DEMOS models. This file serves as the main entry point for defining and sequencing the various simulation models used in the demographic modeling process. Each model is defined as an Orca step, using the `@orca.step` decorator. This allows for modular and flexible model integration within the simulation framework. Each orca step is defined as a function that takes the current state of the agents as input and returns the updated state of the agents after the model has been run. In running each model, the orca step calls different utility functions that are defined in different scripts within the `utils` folder, and which serve as helper functions to pre-process or post-process the data that is used in the model. For example, the Aging model is defined as follows:
```
@orca.step("aging_model")
def aging_model(persons, households):
   # Fetching data and preparing for the model
   persons_df = persons.local
   households_df = households.local
   # Running the model (in this case, incrementing the age of the persons table)
   persons_df["age"] += 1
   # Post-processing the data after the model has been run
   persons_df, households_df = aggregate_household_data(persons_df, households_df)
   # Updating simulation tables
   orca.get_table("persons").update_col("age", persons_df["age"])
   for column in ["age_of_head", "hh_age_of_head", "hh_children", "gt55", "hh_seniors"]:
       orca.get_table("households").update_col(column, households_df[column])
```
Enabling/disabling each of the DEMOS models can be done by commenting/uncommenting the name of the orca step corresponding to the model in the `demo_models` list.
Similarly, the order of the models can be changed by changing the order of the orca steps in the `demo_models` list.


## datasources.py
`datasources.py` is the central hub for data management in the simulation. It handles loading and preparing input data, setting up Orca tables for various entities, and initializing necessary metadata. This file serves as the main entry point for adding any tables that will be accessed by the model during simulation runs.


## variables.py


The `variables.py` file defines derived variables and associated columns used throughout the simulation. Its key purpose is to define the different variables that are used in the simulation or derive new variables from the results of the simulation. All variables are defined as orca decorated functions, allowing them to interact and be used within the simulation.
The model specifications defined in the `.yaml` files within the `configs` folder are used to specify the variable naming conventions for the variables that are used in the simulation.
For example, the following is a simple variable that creates a male gender indicator for persons:
```
@orca.column('persons')
def gender1(persons):
   p = persons.to_frame(columns=['sex'])
   return p.eq(1).astype(int)
```
The underlying logic defining each of the variables in `variables.py` depends on the data schema of the input data as well as the naming conventions that are used in the model specifications. As a result, the orca columns should be given names consistent with variables names in the model specification and their underlying logic should be modified if the input data schema is different from the default input schema.


## indicators.py
The `indicators.py` file is responsible for processing simulation data, creating visualizations, and exporting various indicators and metadata to facilitate analysis and reporting of simulation results.


## simulate.py
The `simulate.py` file is the main entry point for running simulations. It handles command-line arguments, initializes the simulation environment, and executes the simulation steps. [Section II](#ii-setup-guide) describes how to set up the development environment and run the simulation.


## data folder
The data folder is structured as follows:
```
data/
├── custom_mpo_{region_code}_model_data.h5
├── relmap_{region_code}.csv
├── calibration_data/
│   ├── income_growth_rates_{region_code}.csv
│   ├── births_over_time_obs_{region_code}.csv
│   ├── mortalities_over_time_obs_{region_code}.csv
│   ├── hsizec_ct_{region_code}.csv
│   ├── divorces_over_time_obs_{region_code}.csv
│   ├── enrollment_over_time_obs_{region_code}.csv
│   ├── gender_over_time_obs_{region_code}.csv
│   ├── marrital_status_over_time_obs_{region_code}.csv
│   ├── pop_over_time_obs_{region_code}.csv
│   ├── households_over_time_obs_{region_code}.csv
│   ├── marriages_over_time_obs_{region_code}.csv
│   └── income_growth_rates_{region_code}.csv
├── asim_skims/
│   └── skims_mpo_{region_code}.omx
└── school_data/
    ├── blocks_school_districts_{base_year}_{region_code}.csv
    ├── schools_{base_year}_{region_code}.csv
    └── geoid_to_zone_{region_code}.csv

```
- `custom_mpo_{region_code}_model_data.h5`: This file contains the model data for the region. It is a HDF5 file that contains the following tables:
  - `households`: This table contains the household data.
  - `persons`: This table contains the person data.
  - `blocks`: This table contains the block data.
  - `jobs`: This table contains the job data.
  - `schools`: This table contains the school data.
- `rel_map_{region_code}.csv`: This file contains the household role mapping structure after mortality model

The data folder also contains the following subfolders:

## calibration_data subfolder

The `calibration_data` subfolder within the data folder contains files and scripts used for calibrating the DEMOS models. This includes files with historical and forecast controls used for calibrating the ASC of the DEMOS models:
- `income_growth_rates_{region_code}.csv`: This file contains the income growth rates for the region.
- `births_over_time_obs_{region_code}.csv`: This file contains the observed births over time for the region.
- `mortalities_over_time_obs_{region_code}.csv`: This file contains the observed mortalities over time for the region.
- `hsizec_ct_{region_code}.csv`: This file contains the observed household size distribution for the region.
- `divorces_over_time_obs_{region_code}.csv`: This file contains the observed divorces over time for the region.
- `enrollment_over_time_obs_{region_code}.csv`: This file contains the observed enrollment over time for the region.
- `gender_over_time_obs_{region_code}.csv`: This file contains the observed gender distribution over time for the region.
- `marrital_status_over_time_obs_{region_code}.csv`: This file contains the observed marital status distribution over time for the region.
- `pop_over_time_obs_{region_code}.csv`: This file contains the observed population distribution over time for the region.
- `households_over_time_obs_{region_code}.csv`: This file contains the observed household distribution over time for the region.
- `marriages_over_time_obs_{region_code}.csv`: This file contains the observed marriages over time for the region.

This subfolder is crucial for ensuring the models accurately reflect real-world demographic trends and behaviors specific to the region being modeled.

## asim_skims subfolder

The `asim_skims` subfolder within the data folder contains skims data used in any models depending on accessibility variables (e.g.: mandatory location choice models). Data is in `.omx` format. Note that none of the DEMOS model requires the use of the skims data, so these files could be ignored if only DEMOS models are being run.

## school_data subfolder

The `school_data` subfolder within the data folder contains data relevant the the school location assignment step. This data includes:
 - `blocks_school_districts_{base_year}_{region_code}.csv`: This file contains the block-to-school district matching for the region.
 - `schools_{base_year}_{region_code}.csv`: This file contains the school data for the region, including school enrollment capacities.
 - `geoid_to_zone_{region_code}.csv`: This file contains the block-to-taz matching for the region, helpful for assigning school TAZ indicator.

This data is used in education-related models, such as school enrollment predictions and household location choices based on school proximity and quality within the modeled region.

## configs folder


The `configs` folder is a crucial component of the simulation framework, primarily housing the model specifications that define the type, structure, and parameters of various models used in the simulation. These specifications are located in YAML files, each corresponding to a specific model (e.g., `birth_model.yaml` for the birth model), and they outline the variables, coefficients, and other parameters that the models utilize during execution. These model specifications are based on `urbansim-templates`. For more details on the meaning and definition of parameters in the model specifications configs, please refer to the `urbansim-templates`[https://github.com/UDST/urbansim_templates] documentation.
The variable naming conventions in the model specifications must align with the variable definitions specified in `variables.py`.

These files are located in region-specific subfolders within the `custom` folder, defined by the region FIPS code (e.g., `06197001` for the San Francisco Bay Area).
The DEMOS models specifications files are the following:
   - `birth_model.yaml`: Includes the model specifications for the birth model.
   - `mortality_model.yaml`: Includes the model specifications for the mortality model.
   - `edu_model.yaml`: Includes the model specifications for the education model.
   - `marriage_model.yaml`: Includes the model specifications for the single-to-x model.
   - `cohabitation_model.yaml`: Includes the model specifications for the cohabitation-to-x model.
   - `divorce_model.yaml`: Includes the model specifications for the divorce model.
   - `kids_move_model.yaml`: Includes the model specifications for the kids move model.
   - `demos_employed-to-x.yaml`: Includes the model specifications for employment status change when agents are currently employed..
   - `demos_unemployed-to-x.yaml`: Includes the model specifications for employment status change when agents are currently unemployed.


Other models that are needed in simulation but are not part of DEMOS model suite are location choice models. They are located in the same folder as the DEMOS models specifications files, and can be identified as follows:
   - `hlcm*.yaml`: Includes the model specifications for the household location choice models, either for the entire region or for specific counties and population segments.
   - `elcm*.yaml`: Includes the model specifications for the employment location choice models, either for the entire region or for specific counties and population segments.
   - `rdplcm*.yaml`: Includes the model specifications for the retail location choice models, either for the entire region or for specific counties and population segments.
   - `wlcm*.yaml`: Includes the model specifications for the workplace location choice models, either for the entire region or for specific counties and population segments.


## II. Setup Guide


This repository contains only code and configuration/setup files necessary


1. Install Anaconda
It is recommended to create a new development environment to isolate the environment for this software from all other installed environments. This is to avoid creating any conflicts with already installed libraries or software. The recommended way to install environments is through Anaconda. Follow this guide to create a new environment and install the dependencies in it.
Install Anaconda by following these instructions for [Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) [MacOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)
[Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)

2. Clone this repository into your local machine.

3. Create a conda environment
Create a conda environment
```
conda create -n DEMOS python=3.8
```
4. Install dependencies
Activate the created conda environment
```
conda activate DEMOS
```

Install the project dependencies.
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

Access the data directory and download the relevant input data within the data subfolder, following the folder structure described above:


7. Activate environment

Activate the created conda environment
```
conda activate DEMOS
```

8. Running Simulation
To run the DEMOS simulation, use the following command structure from the main DEMOS project folder:

```
python simulate.py -- -r <region ID> -i <input_year> -y <forecast year> -f <iter year frequency> -rm --initial_run -s -sl -sg  -t <skims source> -b -o
```
The following are all the available arguments for the simulate.py script:

```
-r REGION_CODE, --region_code REGION_CODE
                        Region FIPS code (e.g., 06197001 for San Francisco Bay Area)
-i INPUT_YEAR, --input_year INPUT_YEAR
                        Base year of input data
-y YEAR, --year YEAR    Forecast year to simulate to
-f FREQ_INTERVAL, --freq_interval FREQ_INTERVAL
                        Intra-simulation frequency interval for outputs
-rm, --random_matching  Use random matching in the marriage model
--initial_run           Generate calibration/validation charts
-s RANDOM_SEED, --random_seed RANDOM_SEED
                        Value to set as random seed
-sl, --single_level_lcms
                        Run with single-level Location Choice Models (LCMs)
-sg, --segmented        Run with segmented Location Choice Models (LCMs)
-t TRAVEL_MODEL, --travel_model TRAVEL_MODEL
                        Source of skims data (e.g., beam, polaris)
-b CAPACITY_BOOST, --capacity_boost CAPACITY_BOOST
                        Value to multiply capacities during simulation
-o OUTPUT_FNAME, --output_fname OUTPUT_FNAME
                        Custom output file name
```

Here's a basic example using only the DEMOS-relevant options:

```
python simulate.py -r 06197001 -i 2010 -y 2020 -f 2 -rm
```

This command runs the simulation for the San Francisco Bay Area (region code 06197001), starting from the base year 2010 and forecasting to 2020, with outputs every 2 years. It uses random matching in the marriage model.

For more advanced usage, including additional parameters:

```
python simulate.py -r 06197001 -i 2010 -y 2020 -f 2 -rm -s 42 -sg -o custom_output -b 1.2
```
This command includes all the DEMOS options from the first example, and adds several advanced features. It sets a random seed of 42 for reproducibility, uses segmented Location Choice Models (LCMs) for land use models, specifies a custom output filename, and applies a capacity boost of 1.2 to account for potential growth scenarios. This command will direct the software to the correct input data and configuration files and run the simulation based on the specified parameters. The analyst can then use the saved simulation results to run any additional analyses necessary.

9. Simulation results
The DEMOS simulation will produce an output file containing the evolved synthetic population throughout the simulation period. This file, named 'model_data_{forecast_year}.h5', contains the synthetic population of each simulated year. It follows the structure of the input `h5` file.


# Copyright Notice

DEMographic MicrOSimulation (DEMOS) Copyright (c) 2025, UrbanSim Inc.  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
