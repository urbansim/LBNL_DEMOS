# DEMOS: Demographic Microsimulator


This repository contains code that runs simulations for the DEMOS model suite, as fully integrated with Urbansim models. The current DEMOS model suite is implemented for the San Francisco Bay Area and the Austin Area. Several demographic models are implemented along with  Urbansim models. The details about the implemented DEMOS model is provided below: 


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




1. Clone this repository into your local machine


2. Create Environment




3. Run the local setup


```
python3 setup.py develop
```


4. Install Anaconda
It is recommended to create a new development environment to isolate the environment for this software from all other installed environments. This is to avoid creating any conflicts with already installed libraries or software. The recommended way to install environments is through Anaconda. Follow this guide to create a new environment and install the dependencies in it.
Install Anaconda by following these instructions for [Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) [MacOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)
[Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)


Create a conda environment
```
conda create -n DEMOS python=3.8
```
5. Install dependencies
Activate the created conda environment
```
conda activate DEMOS
```


You might need to install or update additional build tools.
For Windows:
In your terminal window, run:
```
conda install -c anaconda libpython m2w64-toolchain
```
For Linux/macOS
In your terminal window, run:
```
apt-get update


apt-get install -y gcc libhdf5-serial-dev
```


Install the project requirements.
```
pip install -r requirements.txt
```
If you run into any issues in setting up your environment, please submit an issue in this repository.


6. Download Relevant Data


Before running the simulation framework, create a data folder within the main project directory by typing the following in your terminal:


```
cd DEMOS


mkdir data
```


Access the data directory and download the relevant input data using the following:


- SF Bay Area
```
wget <INPUT_POPULATION_FILEPATH> # Download the input population file for the region
wget -O bay_area_skims.csv.gz <SKIMS_ILEPATH> # Download the skims file for the region and give it the name bay_area_skims.csv.gz
```


7. Activate environment


Activate the created conda environment
```
conda activate DEMOS_ENV
```
8. Preprocess SKIMS
Before running any simulations, it's necessary to preprocess the skims files downloaded in Step 5 for your relevant region. To process the skims file, first activate your environment using the following command:
```
conda activate DEMOS
```
Access the main DEMOS project folder and running the skims processing file using the following command:
```
cd demos_urbansim
python process_skims.py
```


9. Running Simulation


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


10. Simulation results
The DEMOS simulation will produce the following sets of data and results:
 - A synthetic population file containing the evolved synthetic population throughout the simulation years in `h5` format.


The file can be used to generate summary statistics on population characteristics for each of the simulated years.

