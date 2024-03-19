import math
import os
import time
import warnings
from operator import index
from typing import Sized

warnings.filterwarnings("ignore")

import indicators
import numpy as np
import orca
import pandana as pdna
import pandas as pd
import stopwatch
import utils
import yaml
from google.cloud import storage
from scipy.spatial.distance import cdist
from urbansim.developer import developer

# import demo_models
from urbansim.models import GrowthRateTransition, transition

from urbansim_templates import modelmanager as mm
from urbansim_templates.models import BinaryLogitStep, OLSRegressionStep

# modelmanager.initialize()

print("importing models for region", orca.get_injectable("region_code"))

# -----------------------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------------------


@orca.step("status_report")
def status_report(year):
    print("------------------------------------------------")
    print("STATUS REPORT")
    for tbl in ["households", "jobs", "residential_units"]:
        df = orca.get_table(tbl)
        df = df.local
        unplaced = df[df["block_id"] == "-1"]
        print("------------------------------------------------")
        print(tbl, "unplaced: ", len(unplaced.index))
        if len(unplaced.index) > 0:
            print("Unique values per variable in unplaced agents:")
            for var in df.columns:
                print("         ", var, ": ", unplaced[var].unique())
    print("------------------------------------------------")

@orca.step()
def build_networks(blocks, block_groups, nodes, edges):
    nodes, edges = nodes.local, edges.local
    print("Number of nodes is %s." % len(nodes))
    print("Number of edges is %s." % len(edges))
    nodes = nodes.set_index("id")
    net = pdna.Network(
        nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
    )

    precompute_distance = 1000
    print("Precomputing network for distance %s." % precompute_distance)
    print("Network precompute starting.")
    net.precompute(precompute_distance)
    print("Network precompute done.")

    blocks = blocks.local
    blocks["node_id"] = net.get_node_ids(blocks["x"], blocks["y"])
    orca.add_column("blocks", "node_id", blocks["node_id"])

    block_groups = block_groups.to_frame(["x", "y"])
    block_groups["node_id"] = net.get_node_ids(block_groups["x"], block_groups["y"])
    orca.add_column("block_groups", "node_id", block_groups["node_id"])

    orca.add_injectable("net", net)

# -----------------------------------------------------------------------------------------
# DEMOS
# -----------------------------------------------------------------------------------------

@orca.step("household_stats")
def household_stats(persons, households):
    """Function to print the number of households
    from both the households and pers

    Args:
        persons (DataFrame): Pandas DataFrame of the persons table
        households (DataFrame): Pandas DataFrame of the households table
    """
    # Retrieve data from tables
    persons_table = orca.get_table("persons").local
    households_table = orca.get_table("households").local

    # Unique household IDs from persons table
    unique_households_persons = persons_table["household_id"].unique()
    # Unique household IDs from households table
    unique_households_households = households_table.index.unique()

    # Print statements
    print(f"Households size from persons table: {unique_households_persons.shape[0]}")
    print(f"Households size from households table: {unique_households_households.shape[0]}")

    # Households in one table but not the other
    households_diff_ph = set(unique_households_persons) - set(unique_households_households)
    households_diff_hp = set(unique_households_households) - set(unique_households_persons)
    print(f"Households in households table not in persons table: {len(households_diff_ph)}")
    print(f"Households in persons table not in households table: {len(households_diff_hp)}")

    # Households with NA persons
    print(f"Households with NA persons: {households_table['persons'].isna().sum()}")

    # Duplicated households and persons
    print(f"Duplicated households: {households_table.index.has_duplicates}")
    print(f"Duplicated persons: {persons_table.index.has_duplicates}")

    # Adding new columns to persons_df based on conditions
    persons_table["relate_0"] = np.where(persons_table["relate"] == 0, 1, 0)
    persons_table["relate_1"] = np.where(persons_table["relate"] == 1, 1, 0)
    persons_table["relate_13"] = np.where(persons_table["relate"] == 13, 1, 0)

    # Grouping and aggregating data
    persons_df_sum = persons_table.groupby("household_id").agg(
        relate_1=("relate_1", sum),
        relate_13=("relate_13", sum),
        relate_0=("relate_0", sum)
    )

    # Printing households with multiple relationships
    print("Households with multiple 0: ", (persons_df_sum["relate_0"] > 1).sum())
    print("Households with multiple 1: ", (persons_df_sum["relate_1"] > 1).sum())
    print("Households with multiple 13: ", (persons_df_sum["relate_13"] > 1).sum())
    print("Households with 1 and 13: ", ((persons_df_sum["relate_1"] * persons_df_sum["relate_13"]) > 0).sum())

@orca.step("fatality_model")
def fatality_model(persons, households, year):
    """Function to run the fatality model at the persons level.
    The function also updates the persons and households tables,
    and saves the mortalities table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
    """
    # Retrieve and update 'persons' table
    persons_df = orca.get_table("persons").local
    persons_df["dead"] = -99
    orca.add_table("persons", persons_df)

    # Retrieve and calibrate the mortality model
    mortality = mm.get_step("mortality")
    observed_fatalities_df = orca.get_table("observed_fatalities_data").to_frame()
    target_count = observed_fatalities_df[observed_fatalities_df["year"] == year]["count"].iloc[0]
    fatality_list = utils.calibrate_model(mortality, target_count)

    # Print predicted and observed fatalities
    predicted_fatalities = fatality_list.sum()
    print(f"{predicted_fatalities} predicted fatalities")
    print(f"{target_count} observed fatalities")

    # Update households and persons tables
    utils.remove_dead_persons(
        orca.get_table("persons"), 
        orca.get_table("households"), 
        fatality_list, 
        year
    )

    # Update or create the mortalities table
    mortalities_df = orca.get_table("mortalities").to_frame()
    mortalities_new_row = pd.DataFrame(
        {"year": [year], "count": [predicted_fatalities]}
    )
    mortalities_df = (pd.concat([mortalities_df, mortalities_new_row], ignore_index=True) 
                    if not mortalities_df.empty else mortalities_new_row)
    orca.add_table("mortalities", mortalities_df)

@orca.step("update_income")
def update_income(persons, households, year):
    """
    Updates the income for both persons and households based on the current year's income rates.

    This function processes person and household data to update personal earnings
    and total household income according to the latest income rates by county.

    Args:
        person_table (DataFrameWrapper): Wrapper for the persons table.
        household_table (DataFrameWrapper): Wrapper for the households table.
        simulation_year (int): The current year of simulation.

    Returns:
        None: Directly updates the Orca tables for persons and households.
    """
    # Load data from Orca tables
    person_data = orca.get_table("persons").local
    person_local_columns = orca.get_injectable("persons_local_cols")

    household_data = orca.get_table("households").local
    household_local_columns = household_data.columns

    # Load income rate adjustments for the current year
    annual_income_rates = orca.get_table("income_rates").to_frame()

    annual_income_rates = annual_income_rates[annual_income_rates["year"] == year]

    # Prepare county IDs from household data for merging
    county_ids = household_data[["lcm_county_id"]].copy()

    # Merge person data with household county IDs and income rates
    person_data = person_data.reset_index().merge(
        county_ids.reset_index(), on="household_id").set_index("person_id")
    person_data = person_data.reset_index().merge(
        annual_income_rates, on="lcm_county_id").set_index("person_id")

    # Update personal earnings based on the county-specific rate
    person_data["earning"] *= (1 + person_data["rate"])

    # Aggregate new household incomes
    updated_household_income = person_data.groupby("household_id").agg(
        total_income=("earning", "sum"))

    # Update household data with new income totals
    household_data.update(updated_household_income)
    household_data["income"] = household_data["income"].astype(int)

    # Update Orca tables with modified person and household data
    orca.add_table("persons", person_data[person_local_columns])
    orca.add_table("households", household_data[household_local_columns])

@orca.step("update_age")
def update_age(persons, households):
    """
    This function updates the age of the persons table and
    updates the age of the household head in the household table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None. Updates the Orca tables in place.
    """
    print("AGE")
    persons_df = persons.local
    persons_df["age"] += 1
    households_df = households.to_frame(columns=["age_of_head", "hh_age_of_head"])
    households_df["age_of_head"] += 1
    households_df["hh_age_of_head"] = np.where(
        households_df["age_of_head"] < 35,
        "lt35",
        np.where(households_df["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["person"] = 1
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    households_stats = persons_df.groupby(["household_id"]).agg(
        children=("child", "sum"), seniors=("senior", "sum"), size=("person", "sum")
    )
    households_stats["hh_children"] = np.where(
        households_stats["children"] > 0, "yes", "no"
    )
    households_stats["gt55"] = np.where(households_stats["seniors"] > 0, 1, 0)
    households_stats["hh_seniors"] = np.where(
        households_stats["seniors"] > 0, "yes", "no"
    )
    orca.get_table("households").update_col("age_of_head", households_df["age_of_head"])
    orca.get_table("households").update_col(
        "hh_age_of_head", households_df["hh_age_of_head"]
    )
    orca.get_table("households").update_col(
        "hh_children", households_stats["hh_children"]
    )
    orca.get_table("households").update_col("gt55", households_stats["gt55"])
    orca.get_table("households").update_col(
        "hh_seniors", households_stats["hh_seniors"]
    )
    orca.get_table("persons").update_col("age", persons_df["age"])

@orca.step("education_model")
def education_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None. Updates the Orca persons table.
    """
    # Add temporary variable
    persons_df = persons.local
    persons_df["stop"] = -99
    orca.add_table("persons", persons_df)

    # Run the education model
    edu_model = mm.get_step("education")
    edu_model.run()
    student_list = edu_model.choices.astype(int)

    # Update student status
    persons_df = utils.update_education_status(persons, student_list, year)

    orca.get_table("persons").update_col("edu", persons_df["edu"])
    orca.get_table("persons").update_col("student", persons_df["student"])

@orca.step("laborforce_participation_model")
def laborforce_participation_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Add temporary variable
    persons_df = orca.get_table("persons").local
    persons_df["stay_out"] = -99
    persons_df["leaving_workforce"] = -99
    orca.add_table("persons", persons_df)
    
    #TODO: DOUBLE CHECK DEFINITIONS HERE
    persons_df = orca.get_table("persons").local
    observed_stay_unemployed = orca.get_table("observed_entering_workforce").to_frame()
    observed_exit_workforce = orca.get_table("observed_exiting_workforce").to_frame()
    workforce_eligible_count = len(persons_df[(persons_df["worker"]==0) & (persons_df["age"]>=18)])

    # Entering labor force
    target_unemployed_yearly_share = observed_stay_unemployed[observed_stay_unemployed["year"]==year]["share"]
    target_unemployed_yearly_count = round(target_unemployed_yearly_share * target_unemployed_yearly_share)
    in_workforce_model = mm.get_step("enter_labor_force")
    stay_unemployed_list = utils.calibrate_model(in_workforce_model, target_unemployed_yearly_count)

    # Exit labor force model
    target_exit_workforce_yearly_share = observed_exit_workforce[observed_exit_workforce["year"]==year]["share"]
    target_exit_workforce_yearly_count = round(target_exit_workforce_yearly_share * workforce_eligible_count)
    out_workforce_model = mm.get_step("exit_labor_force")
    exit_workforce_list = utils.calibrate_model(out_workforce_model, target_exit_workforce_yearly_count)
    
    # Update labor status
    utils.update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year)

@orca.step("birth")
def birth_model(persons, households, year):
    """
    Function to run the birth model at the household level.
    The function updates the persons table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    #AUTOGENERATION OF VARIABLE NOT WORKING
    #REF: https://udst.github.io/urbansim_templates/model-steps.html#binary-logit

    print("HERE")
    households_df = orca.get_table("households").local
    households_df["birth"] = -99
    orca.add_table("households", households_df)
    households_df = orca.get_table("households").local
    breakpoint()
    persons_df = persons.to_frame(columns=["sex", "age", "household_id", "relate"])
    observed_births = orca.get_table("observed_births_data").to_frame()
    yearly_observed_births = observed_births[observed_births["year"]==year]["count"]
    print("HERE")
    eligible_households = utils.get_eligible_households(persons_df)
    breakpoint()
    print("HERE")
    # Run model
    birth_model = mm.get_step("birth_model")
    birth_model.filters = "index in " + eligible_households
    birth_model.out_filters = "index in " + eligible_households

    birth_list = utils.calibrate_model(birth_model, yearly_observed_births)

    print(yearly_observed_births.sum(), " target")
    print(birth_list.sum(), " predicted")

    utils.update_birth(persons, households, birth_list)

    # Updating predictions table
    btable_df = orca.get_table("btable").to_frame()
    if btable_df.empty:
        btable_df = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })
    else:
        btable_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })

        btable_df = pd.concat([btable_df, btable_df_new], ignore_index=True)
    orca.add_table("btable", btable_df)


@orca.step("kids_moving_model")
def kids_moving_model(persons, households):
    """
    Running the kids moving model and updating household
    stats.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_df["kid_moves"] = -99
    orca.add_table("persons", persons_df)

    kids_moving_model = mm.get_step("kids_move")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)

    utils.update_households_after_kids(persons, households, kids_moving)

@orca.step("households_reorg")
def households_reorg(persons, households, year):
    """Runs the household reorganization models,
    more specifically: Single-to-X, Married-to-X,
    Cohabitation-to-X


    Args:
        persons (Orca table): Persons orca table
        households (Orca table): Households orca table
        year (int): Simulation year
    """
    # MARRIAGE MODEL
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns

    marriage_model = orca.get_injectable("marriage_model")
    marriage_coeffs = pd.DataFrame(marriage_model["model_coeffs"])
    marriage_variables = pd.DataFrame(marriage_model["spec_names"])
    model_columns = marriage_variables[0].values.tolist()
    vars = persons_local_cols + model_columns
    persons_df = persons.to_frame(columns=vars)
    # get persons cohabitating and heads of their households
    COHABS_PERSONS = persons_df["relate"] == 13
    cohab_persons_df = persons_df.loc[COHABS_PERSONS].copy()
    COHABS_HOUSEHOLDS = cohab_persons_df["household_id"].unique()
    COHABS_HEADS = (persons_df["household_id"].isin(COHABS_HOUSEHOLDS)) & (persons_df["relate"] == 0)
    cohab_heads_df = persons_df.loc[COHABS_HEADS].copy()
    all_cohabs_df = pd.concat([cohab_heads_df, cohab_persons_df])
    all_cohabs_df["cohab"] = 1
    # Get Single People
    SINGLE_COND = (persons_df["MAR"] != 1) & (persons_df["age"] >= 15)
    single_df = persons_df.loc[SINGLE_COND].copy()
    data = single_df.join(all_cohabs_df[["cohab"]], how="left")
    data = data.loc[data["cohab"] != 1].copy()
    data.drop(columns="cohab", inplace=True)
    data = data.loc[:, model_columns].copy()
    ###############################################################
    # print("Running marriage model...")
    marriage_list = utils.simulation_mnl(data, marriage_coeffs)

    random_match = orca.get_injectable("random_match")
    ## ------------------------------------
    
    # DIVORCE MODEL
    households_df = orca.get_table("households").local
    households_df["divorced"] = -99
    orca.add_table("households", households_df)
    persons_df = orca.get_table("persons").local
    ELIGIBLE_HOUSEHOLDS = list(persons_df[(persons_df["relate"].isin([0, 1])) & (persons_df["MAR"] == 1)]["household_id"].unique().astype(int))
    sizes = (persons_df[persons_df["household_id"].isin(ELIGIBLE_HOUSEHOLDS)& (persons_df["relate"].isin([0, 1]))].groupby("household_id").size())
    ELIGIBLE_HOUSEHOLDS = sizes[(sizes == 2)].index.to_list()
    divorce_model = mm.get_step("divorce")
    list_ids = str(ELIGIBLE_HOUSEHOLDS)
    divorce_model.filters = "index in " + list_ids
    divorce_model.out_filters = "index in " + list_ids

    divorce_model.run()
    divorce_list = divorce_model.choices.astype(int)        
    #########################################
    # COHABITATION_TO_X Model
    persons_df = persons.local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    hh_local_cols = orca.get_table("households").local_columns
    
    cohabitation_model = orca.get_injectable("cohabitation_model")
    cohabitation_coeffs = pd.DataFrame(cohabitation_model["model_coeffs"])
    cohabitation_variables = pd.DataFrame(cohabitation_model["spec_names"])
    model_columns = cohabitation_variables[0].values.tolist()
    vars = np.unique(np.array(hh_local_cols + model_columns))
    persons_df = persons.local
    ELIGIBLE_HOUSEHOLDS = (
        persons_df[(persons_df["relate"] == 13) & \
                   (persons_df["MAR"]!=1) & \
                   ((persons_df["age"]>=15))]["household_id"].unique().astype(int)
    )

    data = (
        households.to_frame(columns=model_columns)
        .loc[ELIGIBLE_HOUSEHOLDS, model_columns]
    )

    # Run Model
    cohabitate_x_list = utils.simulation_mnl(data, cohabitation_coeffs)
    
    ######### UPDATING
    print("Restructuring households:")
    print("Cohabitations..")
    utils.update_cohabitating_households(persons, households, cohabitate_x_list)
    print_household_stats()
    
    print("Marriages..")
    utils.update_married_households_random(persons, households, marriage_list)
    print_household_stats()
    utils.fix_erroneous_households(persons, households)
    print_household_stats()
    
    print("Divorces..")
    utils.update_divorce(divorce_list)
    print_household_stats()
    
    marrital = orca.get_table("marrital").to_frame()
    persons_df = orca.get_table("persons").local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    persons_df["member_id"] = persons_df.groupby("household_id")["relate"].rank(method="first", ascending=True).astype(int)
    orca.add_table("persons", persons_df[persons_local_columns])
    if marrital.empty:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        marrital = pd.DataFrame(persons_stats)
        marrital["year"] = year
    else:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        new_marrital = pd.DataFrame(persons_stats)
        new_marrital["year"] = year
        marrital = pd.concat([marrital, new_marrital])
    orca.add_table("marrital", marrital)

def print_household_stats():
    """
    Function to print the number of households from both the households and persons tables.
    """
    persons_table = orca.get_table("persons").local
    households_table = orca.get_table("households").local

    # Printing households size from different tables
    print("Households size from persons table:", persons_table["household_id"].nunique())
    print("Households size from households table:", households_table.index.nunique())
    print("Persons Size:", persons_table.index.nunique())

    # Missing households
    missing_households = set(persons_table["household_id"].unique()) - set(households_table.index.unique())
    print("Missing hh:", len(missing_households))

    # Calculating relationships
    persons_table["relate_0"] = np.where(persons_table["relate"] == 0, 1, 0)
    persons_table["relate_1"] = np.where(persons_table["relate"] == 1, 1, 0)
    persons_table["relate_13"] = np.where(persons_table["relate"] == 13, 1, 0)
    
    persons_df_sum = persons_table.groupby("household_id").agg(
        relate_1=("relate_1", "sum"),
        relate_13=("relate_13", "sum"),
        relate_0=("relate_0", "sum")
    )

    # Printing statistics about households
    print("Households with multiple 0:", (persons_df_sum["relate_0"] > 1).sum())
    print("Households with multiple 1:", (persons_df_sum["relate_1"] > 1).sum())
    print("Households with multiple 13:", (persons_df_sum["relate_13"] > 1).sum())
    print("Households with 1 and 13:", ((persons_df_sum["relate_1"] * persons_df_sum["relate_13"]) > 0).sum())

# -----------------------------------------------------------------------------------------
# MLCM
# -----------------------------------------------------------------------------------------

@orca.step("job_sector")
def job_sector_assignment(persons):
    rates = orca.get_injectable("job_sector_rates")
    persons_local_columns = orca.get_injectable("persons_local_cols").tolist()
    persons_df_local = orca.get_table("persons").local

    # Initialize a column for positive outcomes with zeros
    persons_df = orca.get_table("persons").to_frame(columns=persons_local_columns+["edu_cat", "num_workers_cat"])
    persons['industry'] = 0
    # Assign positive outcomes in chunks based on group size and rate
    for (worker_cat, edu_cat), group in persons_df.groupby(["edu_cat", "num_workers_cat"]):
        group_size = len(group)
        rate = rates.get((worker_cat, edu_cat), None)
        
        if rate is not None:
            # Determine the number of positive outcomes for the group
            positive_outcomes = np.random.binomial(n=group_size, p=rate)
            # Randomly choose individuals within the group to assign positive outcomes
            if positive_outcomes > 0:
                positive_indices = np.random.choice(group.index, size=positive_outcomes, replace=False)
                persons_df.loc[positive_indices, "industry"] = 1

    # breakpoint()
    persons_df.loc[persons_df["num_workers_cat"]==0, "industry"] = -1

    # breakpoint()
    
    orca.add_table("persons", persons_df[persons_local_columns])

@orca.step("work_location")
def work_location(persons):
    """Runs the work location choice model for workers
    in a region. 

    Args:
        persons (Orca table): persons orca table
    """
    print("Starting professional jobs")
    model = mm.get_step('wlcm_js_prof')
    model.run(chooser_batch_size = 100000)
    print("Starting other jobs")
    model = mm.get_step('wlcm_js_other')
    model.run(chooser_batch_size = 100000)   
    persons_work = orca.get_table("persons").to_frame(columns=["work_block_id"])
    persons_work = persons_work.reset_index()
    orca.add_table('work_locations', persons_work.fillna('-1'))


@orca.step("school_location_model")
def school_location_model(persons, households, year):
    """Assigns students to schools based on their grade level and district.

    This function fetches and processes data from Orca tables to assign
    students to appropriate schools within their district.

    Args:
        person_table (Orca Table): Orca table containing person data.
        household_table (Orca Table): Orca table containing household data.
        year (int): The current year of simulation.
    Returns:
        None: Modifies the Orca 'school_locations' table in place.
    """
    # Load necessary data from Orca tables
    person_data = orca.get_table("persons").local
    school_data = orca.get_table("schools").to_frame()
    district_data = orca.get_table("blocks_districts").to_frame()
    household_data = orca.get_table("households").local.reset_index()

    # Filter and prepare student data
    students = utils.extract_students(person_data)
    
    # Map households to districts
    household_district_mapping = household_data[["household_id", "block_id"]].merge(
        district_data, left_on="block_id", right_on="GEOID10")

    # Merge students with their respective household districts
    students_with_districts = students.merge(
        household_district_mapping.reset_index(), on="household_id")
    
    # Filter students based on matching school level and district type
    eligible_students = students_with_districts[
        students_with_districts["STUDENT_SCHOOL_LEVEL"] == students_with_districts["DET_DIST_TYPE"]].copy()

    # Group students for assignment
    student_assignment_groups = utils.create_student_groups(eligible_students)

    # Assign students to schools
    assigned_students = utils.assign_schools(
        student_assignment_groups, district_data, school_data)

    # Create a table of school assignments
    school_assignments = utils.create_results_table(
        eligible_students, assigned_students, year)

    # Update Orca with the new school locations table
    orca.add_table("school_locations", school_assignments[["person_id", "school_id"]])

@orca.step("work_location_stats")
def work_location_stats(persons):
    """Function to generate persons work location
    statistics

    Args:
        persons (Orca table): persons orca table
    Returns:
    None. Prints the work location stats
    """
    # Load the 'persons' table into a DataFrame
    persons_df = orca.get_table("persons").to_frame(columns=["work_block_id", "worker"])

    # Count total number of persons and workers
    total_persons = len(persons_df)
    total_workers = len(persons_df[persons_df["worker"] == 1])

    # Count workers and people with no work location
    workers_no_location = len(persons_df[(persons_df["worker"] == 1) &
                                        (persons_df["work_block_id"] == "-1")])
    people_no_location = len(persons_df[persons_df["work_block_id"] == "-1"])

    # Calculate and print the shares
    share_workers_no_location = (workers_no_location / total_workers
                                if total_workers else 0)
    share_people_no_location = (people_no_location / total_persons
                                if total_persons else 0)

    print("Share of workers with no work location:", share_workers_no_location)
    print("Share of people with no work location:", share_people_no_location)

@orca.step("mlcm_postprocessing")
def mlcm_postprocessing(persons):
    """Assign work and school locations
    to workers and students

    Args:
        persons (Orca table): Orca table of persons
    Returns:
        None. Updates the Orca tables.
    """
    persons_df = orca.get_table("persons").local
    persons_df = persons_df.reset_index()

    geoid_to_zone = orca.get_table("geoid_to_zone").to_frame()
    geoid_to_zone["work_block_id"] = geoid_to_zone["GEOID10"].copy()
    geoid_to_zone["workplace_taz"] = geoid_to_zone["zone_id"].copy()

    persons_df = persons_df.merge(geoid_to_zone[["work_block_id", "workplace_taz"]], on=["work_block_id"], suffixes=('', '_replace'), how="left")
    persons_df["workplace_taz"] = persons_df["workplace_taz_replace"].copy().fillna("-1")
    persons_df["work_zone_id"] = persons_df["workplace_taz"].copy()
    persons_df = persons_df.set_index("person_id")

    persons_cols = orca.get_injectable("persons_local_cols")

    orca.add_table("persons", persons_df[persons_cols])

# -----------------------------------------------------------------------------------------
# TRANSITION
# -----------------------------------------------------------------------------------------

@orca.step('household_transition')
def household_transition(households, persons, year, metadata):

    linked_tables = {'persons': (persons, 'household_id')}
    if ('annual_household_control_totals' in orca.list_tables()) and ('use_database_control_totals' not in orca.list_injectables()):
        control_totals = orca.get_table('annual_household_control_totals').to_frame()
        full_transition(households, control_totals, 'total', year, 'block_id', linked_tables=linked_tables)
    elif ('household_growth_rate' in orca.list_injectables()) and ('use_database_control_totals' not in orca.list_injectables()):
        rate = orca.get_injectable('household_growth_rate')
        simple_transition(households, rate, 'block_id', set_year_built=True, linked_tables=linked_tables)
    elif 'hsize_ct' in orca.list_tables():
        control_totals = orca.get_table('hsize_ct').to_frame()
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id')
    else:
        control_totals = orca.get_table('hct').to_frame()
        if 'hh_type' in control_totals.columns:
            if control_totals[control_totals.index == year].hh_type.min() == -1:
                control_totals = control_totals[['total_number_of_households']]
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id', linked_tables=linked_tables)
    households_df = orca.get_table('households').local
    households_df.loc[households_df['block_id'] == "-1", 'lcm_county_id'] = "-1"
    households_df.index.rename('household_id', inplace=True)
    persons_df = orca.get_table('persons').local

    orca.add_table('households', households_df)
    orca.add_table('persons', persons_df)

    metadata_df = orca.get_table('metadata').to_frame()
    max_hh_id = metadata_df.loc['max_hh_id', 'value']
    max_p_id = metadata_df.loc['max_p_id', 'value']
    if households_df.index.max() > max_hh_id:
        metadata_df.loc['max_hh_id', 'value'] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata_df.loc['max_p_id', 'value'] = persons_df.index.max()
    orca.add_table('metadata', metadata_df)

@orca.step("job_transition")
def job_transition(jobs, year):
    if ("annual_employment_control_totals" in orca.list_tables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        control_totals = orca.get_table("annual_employment_control_totals").to_frame()
        full_transition(jobs, control_totals, "total", year, "block_id")
    elif ("employment_growth_rate" in orca.list_injectables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        rate = orca.get_injectable("employment_growth_rate")
        simple_transition(jobs, rate, "block_id", set_year_built=True)
    else:
        control_totals = orca.get_table("ect").to_frame()
        if "agg_sector" in control_totals.columns:
            if control_totals[control_totals.index == year].agg_sector.min() == -1:
                control_totals = control_totals[["total_number_of_jobs"]]
        full_transition(jobs, control_totals, "total_number_of_jobs", year, "block_id")
    jobs = orca.get_table("jobs").local
    jobs.loc[jobs["block_id"] == "-1", "lcm_county_id"] = "-1"
    jobs.index.rename("job_id", inplace=True)
    orca.add_table("jobs", jobs)

@orca.step("supply_transition")
def supply_transition(households, residential_units, vacancy):
    agents = len(households)
    agent_spaces = len(residential_units)
    if "residential_vacancy_rate" in orca.list_injectables():
        target_vacancy = orca.get_injectable("residential_vacancy_rate")
    else:
        target_vacancy = vacancy
    target = developer.Developer.compute_units_to_build(
        agents, agent_spaces, target_vacancy
    )
    if target > 0:
        growth_rate = target * 1.0 / agent_spaces
        print("Growth rate implied by target vacancy rate: %s" % growth_rate)
        simple_transition(
            residential_units, growth_rate, "block_id", set_year_built=True
        )

        units = orca.get_table("residential_units").local
        units.index.rename("unit_id", inplace=True)
        units.loc[units["block_id"] == "-1", "lcm_county_id"] = "-1"
        orca.add_table("residential_units", units)
    else:
        print(
            "No new residential units to construct; current vacancy > target vacancy (%s)."
            % vacancy
        )

def full_transition(
    agents,
    ct,
    totals_column,
    year,
    location_fname,
    linked_tables=None,
    accounting_column=None,
    set_year_built=False,
):
    """
    Run a transition model based on control totals specified in the usual UrbanSim way
    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    totals_column : str
        String indicating the agent_controls column to use for totals.
    year : int
        The year, which will index into the controls
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables: dict, optional
        Sets the tables linked to new or removed agents to be updated with
        dict of {'table_name':(DataFrameWrapper, 'link_id')}
    accounting_column : str, optional
        Name of column with accounting totals/quantities to apply toward the
        control. If not provided then row counts will be used for accounting.
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current
        simulation year
    Returns
    -------
    Nothing
    """
    print("Running full transition")
    if "agg_sector" in ct.columns:
        ct["agg_sector"] = ct["agg_sector"].astype("str")
    add_cols = [col for col in ct.columns if col != totals_column]
    add_cols = [col for col in add_cols if col not in agents.local.columns]
    agnt = agents.to_frame(list(agents.local.columns) + add_cols)
    print("Total agents before transition: {}".format(len(agnt)))
    idx_name = agnt.index.name
    if agents.name == "households":
        agnt = agnt.reset_index()
        hh_sizes = agnt["hh_size"].unique()
        updated = pd.DataFrame()
        added = pd.Index([])
        copied = pd.Index([])
        removed = pd.Index([])
        ct["lcm_county_id"] = ct["lcm_county_id"].astype(str)
        max_hh_id = agnt.index.max()
        for size in hh_sizes:
            agnt_sub = agnt[agnt["hh_size"] == size].copy()
            ct_sub = ct[ct["hh_size"] == size].copy()
            tran = transition.TabularTotalsTransition(ct_sub, totals_column, accounting_column)
            updated_sub, added_sub, copied_sub, removed_sub = tran.transition(agnt_sub, year)
            updated_sub.loc[added_sub, location_fname] = "-1"
            if updated.empty:
                updated = updated_sub.copy()
            else:
                updated = pd.concat([updated, updated_sub])

            if added.empty:
                added = added_sub.copy()
            else:
                added = added.append(added_sub)
            if copied.empty:
                copied = copied_sub.copy()
            else:
                copied = copied.append(copied_sub)
            if removed.empty:
                removed = removed_sub.copy()
            else:
                removed = removed.append(removed_sub)
    else:
        tran = transition.TabularTotalsTransition(ct, totals_column, accounting_column)
        updated, added, copied, removed = tran.transition(agnt, year)
    if (len(added) > 0) & (agents.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        max_p_id = metadata.loc["max_p_id", "value"]
        if updated.loc[added, "household_id"].min() < max_hh_id:
            persons_df = orca.get_table("persons").local.reset_index()
            unique_hh_ids = updated["household_id"].unique()
            persons_old = persons_df[persons_df["household_id"].isin(unique_hh_ids)]
            updated = updated.sort_values(["household_id"])
            updated["cumulative_hh_count"] = updated.groupby("household_id").cumcount()
            updated = updated.sort_values(by=["cumulative_hh_count"], ascending=False)
            updated.loc[:,"new_household_id"] = np.arange(updated.shape[0]) + max_hh_id + 1
            updated.loc[:,"new_household_id"] = np.where(updated["cumulative_hh_count"]>0, updated["new_household_id"], updated["household_id"])
            sampled_persons = updated.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
            sampled_persons = sampled_persons.sort_values(by=["cumulative_hh_count"], ascending=False)
            sampled_persons.loc[:,"new_person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
            sampled_persons.loc[:,"person_id"] = np.where(sampled_persons["cumulative_hh_count"]>0, sampled_persons["new_person_id"], sampled_persons["person_id"])
            sampled_persons.loc[:,"household_id"] = np.where(sampled_persons["cumulative_hh_count"]>0, sampled_persons["new_household_id"], sampled_persons["household_id"])
            updated.loc[:,"household_id"] = updated.loc[:, "new_household_id"]
            sampled_persons.set_index("person_id", inplace=True, drop=True)
            updated.set_index("household_id", inplace=True, drop=True)
            persons_local_columns = orca.get_injectable("persons_local_cols")
            orca.add_table("persons", sampled_persons.loc[:,persons_local_columns])

    if agents.name != "households":
        updated.loc[added, location_fname] = "-1"
    if set_year_built:
        updated.loc[added, "year_built"] = year
    updated_links = {}
    if linked_tables:
        for table_name, (table, col) in linked_tables.items():
            print("updating linked table {}".format(table_name))
            updated_links[table_name] = update_linked_table(
                table, col, added, copied, removed
            )
            orca.add_table(table_name, updated_links[table_name])
    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])
    return updated, added, copied, removed

def simple_transition(
    tbl, rate, location_fname, linked_tables={}, set_year_built=False
):
    """
    Run a simple growth rate transition model

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with dict of
        {'table_name':(DataFrameWrapper, 'link_id')}
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    Returns
    -------
    Nothing
    """
    print("Running simple transition with ", rate * 100, "% rate")
    transition = GrowthRateTransition(rate)
    df_base = tbl.to_frame(tbl.local_columns)
    print("%d agents before transition" % len(df_base.index))
    df, added, copied, removed = transition.transition(df_base, None)
    print("%d agents after transition" % len(df.index))
    if (len(added) > 0) & (tbl.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        if added.min() < max_hh_id:
            # reset "added" row IDs so that new rows do not assign
            # IDs of previously removed rows.
            new_max = max(df_base.index.max(), df.index.max())
            new_added = np.arange(len(added)) + new_max + 1
            idx_name = df.index.name
            df["new_idx"] = None
            df.loc[added, "new_idx"] = new_added
            not_added = df["new_idx"].isnull()
            df.loc[not_added, "new_idx"] = df.loc[not_added].index.values
            df.set_index("new_idx", inplace=True, drop=True)
            df.index.name = idx_name
            added = new_added

    df.loc[added, location_fname] = "-1"

    if set_year_built:
        df.loc[added, "year_built"] = orca.get_injectable("year")
    updated_links = {}
    for table_name, (table, col) in linked_tables.items():
        updated_links[table_name] = update_linked_table(
            table, col, added, copied, removed
        )
        orca.add_table(table_name, updated_links[table_name])
    orca.add_table(tbl.name, df)

def update_linked_table(tbl, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.
    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.
    Returns
    -------
    updated : pandas.DataFrame
    """
    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # handle removals
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]
    removed = table.loc[table[col_name].isin(set(removed))]
    if added is None or len(added) == 0:
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat(
        [pd.Series(copied, name=col_name), pd.Series(added, name="temp_id")], axis=1
    )

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={"temp_id": col_name}, inplace=True)

    # index the new rows
    starting_index = max_id + 1
    new_rows.index = np.arange(
        starting_index, starting_index + len(new_rows), dtype=np.int)
    new_rows.index.name = table.index.name

    return pd.concat([table, new_rows])

@orca.step('households_relocation_basic')
def households_relocation_basic(households):
    """
    Running a household relocation model
    """
    #TODO: create injectable for relocation rate
    return simple_relocation(households, .034, "block_id")

def simple_relocation(choosers, relocation_rate, fieldname):
    """Function to run a simple relocation model for choosers
    following a certain rate

    Args:
        choosers (Orca Table): Orca table of the agents deciding to relocate
        relocation_rate (float): Relocation rate for choosers
        fieldname (str): Location field
    Returns:
    None. Updates the Orca choosers table in place.
    """
    print("Total agents: %d" % len(choosers))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))
    print("Assigning for relocation...")
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate * len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname, pd.Series('-1', index=chooser_ids))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))

# -----------------------------------------------------------------------------------------
# DATA EXPORT
# -----------------------------------------------------------------------------------------

@orca.step("generate_outputs")
def generate_outputs(year, base_year, forecast_year, tracts):
    print(
        "Generating outputs for (year {}, forecast year {})...".format(
            year, forecast_year
        )
    )
    if not os.path.exists("runs"):
        os.makedirs("./runs")

    if orca.get_injectable("all_local"):
        return

    if year == base_year:
        indicators.export_indicator_definitions()

    cfg = orca.get_injectable("output_parameters")

    # Layer indicators
    indicators.gen_all_indicators(cfg["output_indicators"], year)

    # Chart indicators
    # if year == forecast_year:
    #     chart_data, geo_small, geo_large = indicators.prepare_chart_data(cfg, year)
    #     # indicators.gen_all_charts(cfg['output_charts'], base_year, forecast_year, chart_data, geo_large)

    # Calibration metrics for pdf report in calibration/microsimulation routine
    if (year == 2018) and (orca.get_injectable("local_simulation") == True):
        indicators.gen_calibration_metrics(tracts)

@orca.step("export_demo_stats")
def export_demo_stats(year, forecast_year):
    """
    Export Demographic Stats tables

    Args:
        year (int): simulation year
        forecast_year (int): final forecast year

    Returns:
        None
    """
    if year == forecast_year:
        for table in ["pop_over_time", "hh_size_over_time", "age_over_time",
                      "edu_over_time", "income_over_time", "kids_move_table",
                      "divorce_table", "marriage_table", "btable",
                      "age_dist_over_time", "pop_size_over_time", 
                      "student_population", "mortalities",
                     "btable_elig", "marrital"]:
            export(table) 

def export(table_name):
    """
    Export the tables

    Args:
        table_name (string): Name of the orca table
    """
    
    region_code = orca.get_injectable("region_code")
    output_folder = orca.get_injectable("output_folder")
    df = orca.get_table(table_name).to_frame()
    csv_name = table_name + "_" + region_code +".csv"
    df.to_csv(output_folder+csv_name, index=False)

# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------

all_local = orca.get_injectable("all_local")
if orca.get_injectable("running_calibration_routine") == False:
    region_code = orca.get_injectable("region_code")

    if not all_local:
        storage_client = storage.Client("swarm-test-1470707908646")
        bucket = storage_client.get_bucket("national_block_v2")

    county_ids = orca.get_table("blocks").county_id.unique()
    rdplcm_segments = ["sf", "mf"]
    hlcm_segments = [
        "own_1p_54less",
        "own_1p_55plus",
        "own_2p_54less",
        "own_2p_55plus",
        "rent_1p_54less",
        "rent_1p_55plus",
        "rent_2p_54less",
        "rent_2p_55plus",
    ]
    elcm_segments = ["0", "1", "2", "3", "4", "5"]

    if orca.get_injectable("calibrated") == True:
        rdplcm_models = []
        hlcm_models = []
        elcm_models = []
        price_models = []
        if orca.get_injectable("multi_level_lcms") == True:
            if orca.get_injectable("segmented_lcms") == True:
                for county_id in county_ids:
                    rdplcm_models += [
                        "rdplcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in rdplcm_segments
                    ]
                    hlcm_models += [
                        "hlcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in hlcm_segments
                    ]
                    elcm_models += [
                        "elcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in elcm_segments
                    ]
                if len(county_ids) > 1:
                    rdplcm_models = [
                        "rdplcm_county_%s_pf" % segment for segment in rdplcm_segments
                    ] + rdplcm_models
                    hlcm_models = [
                        "hlcm_county_%s_pf" % segment for segment in hlcm_segments
                    ] + hlcm_models
                    elcm_models = [
                        "elcm_county_%s_pf" % segment for segment in elcm_segments
                    ] + elcm_models
            else:
                rdplcm_models += [
                    "rdplcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                hlcm_models += [
                    "hlcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                elcm_models += [
                    "elcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                if len(county_ids) > 1:
                    rdplcm_models = ["rdplcm_county_pf"] + rdplcm_models
                    hlcm_models = ["hlcm_county_pf"] + hlcm_models
                    elcm_models = ["elcm_county_pf"] + elcm_models
        else:
            if orca.get_injectable("segmented_lcms") == True:
                rdplcm_models += ["rdplcm_pf_" + segment for segment in rdplcm_segments]
                hlcm_models += ["hlcm_pf_" + segment for segment in hlcm_segments]
                elcm_models += ["elcm_pf_" + segment for segment in elcm_segments]
            else:
                rdplcm_models += ["rdplcm_pf"]
                hlcm_models += ["hlcm_pf"]
                elcm_models += ["elcm_pf"]

        developer_models = ["supply_transition"] + rdplcm_models
        household_models = ["household_transition"] + ["households_relocation_basic"] + hlcm_models
        employment_models = ["job_transition"] + elcm_models
        location_models = rdplcm_models + hlcm_models + elcm_models
        calibrated_folder = orca.get_injectable("calibrated_folder")
        region_type = orca.get_injectable("region_type")
        remote_configs_path = "calibrated_configs/%s/%s" % (
            calibrated_folder,
            region_code,
        )
        if calibrated_folder == "custom":
            remote_configs_path = "calibrated_configs/custom/custom_%s_%s" % (
                region_type,
                region_code,
            )
        local_configs_path = "calibrated_configs"
        if orca.get_injectable("local_simulation") is True:
            local_configs_path = os.path.join(
                local_configs_path, calibrated_folder, region_code
            )
            skim_source = orca.get_injectable("skim_source")
            if os.path.exists(os.path.join("configs", local_configs_path, skim_source)):
                local_configs_path = os.path.join(local_configs_path, skim_source)
        if not os.path.exists("configs/" + local_configs_path):
            os.makedirs("./configs/" + local_configs_path)
        for f in location_models:
            if not all_local:
                print(
                    "Downloading %s config from calibrated_configs/%s"
                    % (f, calibrated_folder)
                )
                blob = bucket.get_blob("%s/%s.yaml" % (remote_configs_path, f))
                blob.download_to_filename(
                    "./configs/%s/%s.yaml" % (local_configs_path, f)
                )
            else:
                if not os.path.exists("./configs/%s/%s.yaml" % (local_configs_path, f)):
                    raise OSError(
                        "No model config found at ./configs/%s/%s.yaml"
                        % (local_configs_path, f)
                    )

        for model in ["value", "rent"]:
            print("Checking if %s configs exist" % model)
            model_name = "repm_residential_%s" % model
            if not all_local:
                blob = bucket.blob("%s/%s.yaml" % (remote_configs_path, model_name))
                if blob.exists():
                    print("Downloading %s" % model_name)
                    blob.download_to_filename(
                        "./configs/%s/%s.yaml" % (local_configs_path, model_name)
                    )
                    price_models += [model_name]
            else:
                if os.path.exists(
                    "./configs/%s/%s.yaml" % (local_configs_path, model_name)
                ):
                    price_models += [model_name]

    else:
        rdplcm_models = ["rdplcm" + segment for segment in rdplcm_segments]
        hlcm_models = ["hlcm" + segment for segment in hlcm_segments]
        elcm_models = ["elcm" + segment for segment in elcm_segments]
        developer_models = ["supply_transition"] + [
            "rdplcm" + str(segment) for segment in range(0, 4)
        ]
        household_models = ["household_transition"] + ["households_relocation_basic"] + ["household_stats"], [
            "hlcm" + str(segment) for segment in range(1, 11)
        ]
        employment_models = ["job_transition"] + [
            "elcm" + str(segment) for segment in range(0, 6)
        ]
        location_models = rdplcm_models + hlcm_models + elcm_models
        price_models = [
            region_code + "_pred_bg_median_rent",
            region_code + "_pred_bg_median_value",
        ]

        if not os.path.exists("configs/estimated_configs"):
            os.makedirs("./configs/estimated_configs")
        for f in location_models:
            if not all_local:
                print("Downloading %s config from estimated_configs" % f)
                blob = bucket.get_blob(
                    "estimated_configs/us/%s/%s.yaml" % (region_code, f)
                )
                blob.download_to_filename("./configs/estimated_configs/%s.yaml" % f)
            else:
                if not os.path.exists("./configs/estimated_configs/%s.yaml" % f):
                    raise OSError(
                        "No model config found at ./configs/estimated_configs/%s.yaml"
                        % f
                    )

    if orca.get_injectable("local_simulation") == True:
        start_of_year_models = ["status_report"]
        demo_models = [
            "update_age",
            # "laborforce_participation_model",
            # "households_reorg",
            # "kids_moving_model",
            "fatality_model",
            "birth_model",
            "education_model",
            "export_demo_stats",
        ]
        pre_processing_steps = price_models + ["build_networks", "generate_outputs", "update_travel_data"]
        export_demo_steps = ["export_demo_stats"]
        household_stats = ["household_stats"]
        school_models = ["school_location_model"]
        end_of_year_models = ["generate_outputs"]
        work_models = ["job_sector", "work_location"]
        mlcm_postprocessing = ["mlcm_postprocessing"]
        update_income = ["update_income"]
        steps_all_years = (
            start_of_year_models
            + demo_models
            # + work_models
            # + school_models
            # + ["work_location_stats"]
            + price_models
            + developer_models
            + household_models
            + employment_models
            + end_of_year_models
            + mlcm_postprocessing
            + export_demo_steps
        )
    else:
        start_of_year_models = [
            "status_report",
            "skim_swapper",
            "scheduled_development_events_model",
        ]
        end_of_year_models = ["adjustment_model", "generate_outputs"]
        pre_processing_steps = (
            ["scenario_definition", "scheduled_development_events_model"]
            + price_models
            + ["build_networks", "skim_swapper", "generate_outputs"]
        )
        steps_all_years = (
            start_of_year_models
            + price_models
            + developer_models
            + household_models
            + employment_models
            + end_of_year_models
        )
    orca.add_injectable("sim_steps", steps_all_years)
    orca.add_injectable("pre_processing_steps", pre_processing_steps)
    orca.add_injectable("export_demo_stats", export_demo_steps)
