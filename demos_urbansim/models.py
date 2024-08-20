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
import yaml
from google.cloud import storage
from scipy.spatial.distance import cdist
from urbansim.developer import developer
from demos_urbansim.demos_utils.utils import (
    increment_ages, update_education_status, simulation_mnl, 
    update_birth_eligibility_count_table, 
    update_births_predictions_table, 
    get_birth_eligible_households, 
    update_birth, update_metadata, update_income,
    update_labor_status,
    update_workforce_stats_tables,
    aggregate_household_labor_variables,
    extract_students, create_student_groups, assign_schools, create_results_table, export_demo_table,
    update_kids_moving_table,
    update_households_after_kids,
    deduplicate_updated_households

)

# import demo_models
from urbansim.models import GrowthRateTransition, transition
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import BinaryLogitStep, OLSRegressionStep

# modelmanager.initialize()

print("importing models for region", orca.get_injectable("region_code"))

# -----------------------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------------------

@orca.step("work_location_stats")
def work_location_stats(persons):
    """Function to generate persons work location
    stats

    Args:
        persons (Orca table): persons orca table
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



@orca.step("add_temp_variables")
def add_temp_variables():
    """Adds temporary variables to the persons and
    households tables.
    """
    persons = orca.get_table("persons").local
    persons.loc[:, "dead"] = -99
    persons.loc[:, "stop"] = -99
    persons.loc[:, "kid_moves"] = -99

    orca.add_table("persons", persons)


@orca.step("remove_temp_variables")
def remove_temp_variables():
    """Removes temporary variables from the persons
    and households tables
    """
    persons = orca.get_table("persons").local
    persons = persons.drop(columns=["dead", "stop", "kid_moves"])
    orca.add_table("persons", persons)


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


def calibrate_model(model, target_count, threshold=0.05):
    model.run()
    predictions = model.choices.astype(int)
    predicted_share = predictions.sum() / predictions.shape[0]
    target_share = target_count / predictions.shape[0]

    error = (predictions.sum() - target_count.sum())/target_count.sum()
    while np.abs(error) >= threshold:
        model.fitted_parameters[0] += np.log(target_count.sum()/predictions.sum())
        model.run()
        predictions = model.choices.astype(int)
        predicted_share = predictions.sum() / predictions.shape[0]
        error = (predictions.sum() - target_count.sum())/target_count.sum()
    return predictions


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
    fatality_list = calibrate_model(mortality, target_count)

    # Print predicted and observed fatalities
    predicted_fatalities = fatality_list.sum()
    print(f"{predicted_fatalities} predicted fatalities")
    print(f"{target_count} observed fatalities")

    # Update households and persons tables
    remove_dead_persons(
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

def identify_dead_households(persons_df):
    persons_df["member"] = 1
    dead_fraction = persons_df.groupby("household_id").agg(
        num_dead=("dead", "sum"), size=("member", "sum")
    )
    return dead_fraction[dead_fraction["num_dead"] == dead_fraction["size"]].index.to_list()

# Mortality model returns a list of 0s representing alive and 1 representing dead
# Then adds that list to the persons table and updates persons and households tables accordingly
def remove_dead_persons(persons, households, fatality_list, year):
    """
    This function updates the persons table from the output of the fatality model.
    Takes in the persons and households orca tables.

    Args:
        persons (DataFramWrapper): DataFramWrapper of persons table
        households (DataFramWrapper): DataFramWrapper of households table
        fatality_list (pd.Series): Pandas Series of fatality list
    """

    houses = households.local
    households_columns = orca.get_injectable("households_local_cols")

    # Pulling the persons data
    persons_df = persons.local
    persons_columns = orca.get_injectable("persons_local_cols")

    persons_df["dead"] = -99
    persons_df["dead"] = fatality_list
    graveyard = persons_df[persons_df["dead"] == 1].copy()

    # HOUSEHOLD WHERE EVERYONE DIES
    dead_households = identify_dead_households(persons_df)
    grave_persons = persons_df[persons_df["household_id"].isin(dead_households)].copy()
    # Drop out of the persons table
    persons_df = persons_df.loc[~persons_df["household_id"].isin(dead_households)]
    # Drop out of the households table
    houses = houses.drop(dead_households)

    ##################################################
    ##### HOUSEHOLDS WHERE PART OF HOUSEHOLD DIES ####
    ##################################################
    dead, alive = persons_df[persons_df["dead"] == 1].copy(), persons_df[persons_df["dead"] == 0].copy()

    #################################
    # Alive heads, Dead partners
    #################################
    # Dead partners, either married or cohabitating
    dead_partners = dead[dead["relate"].isin([1, 13])]  # This will need changed
    # Alive heads
    alive_heads = alive[alive["relate"] == 0]
    alive_heads = alive_heads[["household_id", "MAR"]]
    widow_heads = alive[(alive["household_id"].isin(dead_partners["household_id"])) & (alive["relate"]==0)].copy()
    widow_heads["MAR"].values[:] = 3

    # Dead heads, alive partners
    dead_heads = dead[dead["relate"] == 0]
    alive_partner = alive[alive["relate"].isin([1, 13])].copy()
    alive_partner = alive_partner[
        ["household_id", "MAR"]
    ]  ## Pull the relate status and update it
    widow_partners = alive[(alive["household_id"].isin(dead_heads["household_id"])) & (alive["relate"].isin([1, 13]))].copy()

    widow_partners["MAR"].values[:] = 3  # THIS MIGHT NEED VERIFICATION, WHAT DOES MAR MEAN?

    # Merge the two groups of widows
    widows = pd.concat([widow_heads, widow_partners])[["MAR"]]
    # Update the alive database's MAR values using the widows table
    alive_copy = alive.copy()
    alive.loc[widows.index, "MAR"] = 3
    alive["MAR"] = alive["MAR"].astype(int)


    # Select the households in alive where the heads died
    alive_sort = alive[alive["household_id"].isin(dead_heads["household_id"])].copy()
    alive_sort["relate"] = alive_sort["relate"].astype(int)

    if len(alive_sort.index) > 0:
        alive_sort.sort_values("relate", inplace=True)
        # Restructure all the households where the head died
        alive_sort = alive_sort[["household_id", "relate", "age"]]
        # print("Starting to restructure household")
        # Apply the rez function
        alive_sort = alive_sort.groupby("household_id").apply(rez)

        # Update relationship values and make sure correct datatype is used
        alive.loc[alive_sort.index, "relate"] = alive_sort["relate"]
        alive["relate"] = alive["relate"].astype(int)

    alive["is_relate_0"] = (alive["relate"]==0).astype(int)
    alive["is_relate_1"] = (alive["relate"]==1).astype(int)

    alive_agg = alive.groupby("household_id").agg(sum_relate_0 = ("is_relate_0", "sum"), sum_relate_1 = ("is_relate_1", "sum"))
    
    # Dropping households with more than one head or more than one partner
    alive_agg = alive_agg[(alive_agg["sum_relate_1"]<=1) & (alive_agg["sum_relate_0"]<=1)]
    alive_hh = alive_agg.index.tolist()
    alive = alive[alive["household_id"].isin(alive_hh)]

    alive["person"] = 1
    alive["is_head"] = np.where(alive["relate"] == 0, 1, 0)
    alive["race_head"] = alive["is_head"] * alive["race_id"]
    alive["age_head"] = alive["is_head"] * alive["age"]
    alive["hispanic_head"] = alive["is_head"] * alive["hispanic"]
    alive["child"] = np.where(alive["relate"].isin([2, 3, 4, 14]), 1, 0)
    alive["senior"] = np.where(alive["age"] >= 65, 1, 0)
    alive["age_gt55"] = np.where(alive["age"] >= 55, 1, 0)

    households_new = alive.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    houses.update(households_new)
    houses = houses.loc[alive_hh]

    graveyard_table = orca.get_table("pop_over_time").to_frame()
    if graveyard_table.empty:
        dead_people = grave_persons.copy()
    else:
        dead_people = pd.concat([graveyard_table, grave_persons])

    orca.add_table("persons", alive[persons_columns])
    orca.add_table("households", houses[households_columns])
    orca.add_table("graveyard", dead_people[persons_columns])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

def rez(group):
    """
    Function to change the household head role
    TODO: This needs to become vectorized to make it faster.
    """
    # Update the relate variable for the group
    if group["relate"].iloc[0] == 1:
        group["relate"].iloc[0] = 0
        return group
    if 13 in group["relate"].values:
        group["relate"].replace(13, 0, inplace=True)
        return group

    # Get the maximum age of the household, oldest person becomes head of household
    # Verify this with Juan.
    new_head_idx = group["age"].idxmax()
    # Function to map the relation of new head
    map_func = produce_map_func(group.loc[new_head_idx, "relate"])
    group.loc[new_head_idx, "relate"] = 0
    # breakpoint()
    group.relate = group.relate.map(map_func)
    return group

# TODO refactor this method so it can be cleanly used with any model
# Might need to add some stuff (update more variables) or have a flag to determine which function is calling this
def update_households(alive, dead, old_incomes, subtract=True):
    """
    Function to update the households characteristics, namely income, num of workers, race, and age of head.
    """
    # By default the dead variable represents people dying or leaving a household
    if subtract:
        alive = alive.sort_values("relate")
        dead_aggs = dead.groupby("household_id").agg({"earning": "sum"})

        aggregates = alive.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum", "race_id": "first", "age": "first"}
        )
        # print(list(orca.get_table("households").to_frame().columns))
        # TODO -- REPLACE THESE OPERATIONS BY PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", old_incomes.subtract(dead_aggs["earning"], fill_value=0)
        )
        orca.get_table("households").update_col("workers", aggregates["worker"])
        orca.get_table("households").update_col("race_of_head", aggregates["race_id"])
        orca.get_table("households").update_col("age_of_head", aggregates["age"])
    # In a certain case, the dead variable actually represents living people being added -- What is this case?
    else:
        aggregates = dead.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum"}
        )
        house_df = orca.get_table("households").to_frame(columns=["income", "workers"])
        # TODO -- REPLACE THESE OPERATIONS WITH PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", house_df["income"].add(aggregates["earning"], fill_value=0)
        )
        orca.get_table("households").update_col(
            "workers", house_df["workers"].add(aggregates["worker"], fill_value=0)
        )

    # return households

# Function that takes the head's previous role and returns a function
# that maps roles to new roles based on restructuring
def produce_map_func(old_role):
    """
    Function that uses the relationship mapping in the
    provided table and returns a function that maps
    new household roles.
    """
    # old role is the previous number of the person who has now been promoted to head of the household
    sold_role = str(old_role)
 
    def inner(role):
        rel_map = orca.get_table("rel_map").to_frame()
        if role == 0:
            new_role = 0
        else:
            new_role = rel_map.loc[role, sold_role]
        return new_role

    # Returns function that takes a persons old role and gives them a new one based on how the household is restructured
    return inner

@orca.step("aging_model")
def aging_model(persons, households):
    """
    This function updates the age of the persons table and
    updates the age of the household head in the household table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None. Updates the age related columns in the households and persons tables.
    """
    persons_df = persons.local

    # Increment the age of the persons table
    persons_df, households_ages = increment_ages(persons_df)

    # Update age related columns in the households table
    for column in ["age_of_head", "hh_age_of_head", "hh_children", "gt55", "hh_seniors"]:
        orca.get_table("households").update_col(column, households_ages[column])

    # Update age in the persons table
    orca.get_table("persons").update_col("age", persons_df["age"])

@orca.step("income_model")
def income_model(persons, households, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    # Pulling data, income rates, and county IDs
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    income_rates = orca.get_table("income_rates").to_frame()
    
    persons_df, households_df = update_income(persons_df, households_df, income_rates, year)

    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("households", households_df[households_local_columns])

@orca.step("education_model")
def education_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        year (int): Year of the simulation

    Returns:
        None. Updates the education status of the persons table
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
    persons_df = update_education_status(persons_df, student_list, year)

    # Update the tables with new education values
    orca.get_table("persons").update_col("edu", persons_df["edu"])
    orca.get_table("persons").update_col("student", persons_df["student"])
    
@orca.step("birth_model")
def birth_model(persons, households, year):
    """
    Function to run the birth model at the household level.
    The function updates the persons table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        year (int): simulation year

    Returns:
        None
    """

    households_df = households.local
    households_df["birth"] = -99
    orca.add_table("households", households_df)
    households_df = households.local
    persons_df = persons.local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    btable_df = orca.get_table("btable").to_frame()
    btable_elig_df = orca.get_table("btable_elig").to_frame()
    metadata = orca.get_table("metadata").to_frame()

    eligible_household_ids = get_birth_eligible_households(persons_df, households_df)
    btable_elig_df = update_birth_eligibility_count_table(btable_elig_df, eligible_household_ids, year)

    # Run model
    birth = mm.get_step("birth")
    list_ids = str(eligible_household_ids)
    birth.filters = "index in " + list_ids
    birth.out_filters = "index in " + list_ids

    # Calibrate model
    observed_births = orca.get_table("observed_births_data").to_frame()
    target_count = observed_births[observed_births["year"]==year]["count"]
    birth_list = calibrate_model(birth, target_count)

    print(target_count.sum(), " target")
    print(birth_list.sum(), " predicted")

    # Update persons and households
    persons_df, households_df = update_birth(persons_df, households_df, birth_list)

    # Update births predictions table
    btable_df = update_births_predictions_table(btable_df, year, birth_list)

    # Update metadata
    metadata = update_metadata(metadata, households_df, persons_df)

    # Update tables
    orca.add_table("persons", persons_df.loc[:, persons_local_columns])
    orca.add_table("households", households_df.loc[:, households_local_columns])
    orca.add_table("metadata", metadata)
    orca.add_table("btable_elig", btable_elig_df)
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
    # Pulling data
    persons_df = orca.get_table("persons").local
    persons_df["kid_moves"] = -99
    orca.add_table("persons", persons_df)
    persons_df = orca.get_table("persons").local
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_df = orca.get_table("households").local
    households_local_cols = orca.get_injectable("households_local_cols")
    metadata = orca.get_table("metadata").to_frame()
    kids_moving_table = orca.get_table("kids_move_table").to_frame()
    # Running model
    kids_moving_model = mm.get_step("kids_move")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)
    # Post-processing data
    persons_df, households_df = update_households_after_kids(persons_df, households_df, kids_moving, metadata)
    metadata = update_metadata(metadata, households_df, persons_df)
    kids_moving_table = update_kids_moving_table(kids_moving_table, kids_moving)
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_table("metadata", metadata)
    orca.add_table("kids_move_table", kids_moving_table)

def fix_erroneous_households(persons, households):
    print("Fixing erroneous households")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    # print("Hh size: ", household_df.shape)
    # print("Persons size: ", p_df.shape)
    households_to_drop = p_df[p_df['relate'].isin([1, 13])].groupby('household_id')['relate'].nunique().reset_index()
    households_to_drop = households_to_drop[households_to_drop["relate"]==2]["household_id"].to_list()
    # print("Num hh to be dropped: ", len(households_to_drop))
    household_df = household_df.drop(households_to_drop)
    p_df = p_df[~p_df["household_id"].isin(households_to_drop)]

    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)

def update_married_households_random(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households
    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals
    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()

    if ((relevant["new_mar"] ==1).sum() <= 10) or ((relevant["new_mar"] ==2).sum() <= 10):
        return None

    relevant.sort_values("new_mar", inplace=True)

    relevant = relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")


    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result
    
    min_mar_male = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].shape[0]
    min_mar_female = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].shape[0]

    min_cohab_male = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].shape[0]
    min_cohab_female = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].shape[0]
    
    min_mar = int(min(min_mar_male, min_mar_female))
    min_cohab = int(min(min_cohab_male, min_cohab_female))

    if (min_mar == 0) or (min_mar == 0):
        return None

    female_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].sample(min_mar)
    male_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].sample(min_mar)
    female_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].sample(min_cohab)
    male_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].sample(min_cohab)
    

    female_mar = female_mar.sort_values("age")
    male_mar = male_mar.sort_values("age")
    female_coh = female_coh.sort_values("age")
    male_coh = male_coh.sort_values("age")
    female_mar["number"] = np.arange(female_mar.shape[0])
    male_mar["number"] = np.arange(male_mar.shape[0])
    female_coh["number"] = np.arange(female_coh.shape[0])
    male_coh["number"] = np.arange(male_coh.shape[0])
    married = pd.concat([male_mar, female_mar])
    cohabitate = pd.concat([male_coh, female_coh])
    married = married.sort_values(by=["number"])
    cohabitate = cohabitate.sort_values(by=["number"])

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(by=["household_group", "earning"], ascending=[True, False])
    cohabitate = cohabitate.sort_values(by=["household_group", "earning"], ascending=[True, False])

    cohabitate["household_group"] = (cohabitate["household_group"] + married["household_group"].max() + 1)

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)
    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])].copy()

    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[final[CONDITION_4]["partner"].values, "new_household_id"] = new_household_ids

    # print('Finished Pairing')
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())
    final["hh_new_id"] = np.where(final["stay"].isin([1]), final["household_id"], np.where(final["stay"].isin([0]),final["partner_house"],final["new_household_id"] + current_max_id + 1))

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)]["household_id"].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]

    households_restructuring = p_df.loc[p_df["household_id"].isin(household_ids_reorganized)]

    households_restructuring = households_restructuring.sort_values(by=["household_id", "earning"], ascending=False)
    households_restructuring.loc[households_restructuring.groupby(["household_id"]).head(1).index, "relate"] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_id", "first"),age_of_head=("age", "first"),size=("person", "sum"),workers=("worker", "sum"),hispanic_head=("hispanic_head", "sum"),persons_age_gt55=("age_gt55", "sum"),seniors=("senior", "sum"),children=("child", "sum"),persons=("person", "sum"),)

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["hh_workers"] = np.where(household_agg["workers"] == 0,"none",np.where(household_agg["workers"] == 1, "one", "two or more"),)
    household_agg["hh_age_of_head"] = np.where(household_agg["age_of_head"] < 35,"lt35",np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    p_df["NEW_MAR"] = final["MAR"]
    p_df["MAR"] = np.where(p_df["NEW_MAR"].isna(),p_df["MAR"], p_df["NEW_MAR"])

    new_hh = household_agg.loc[~household_agg.index.isin(household_df.index.unique())].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = "-1"
    household_df = pd.concat([household_df, new_hh])

    # print('Time to run marriage', sp.duration)
    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])

    # print("households size", household_df.shape[0])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)
    
    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        new_married_table = pd.DataFrame(
                [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
                columns=["married", "cohabitated"]
            )
        married_table = pd.concat([married_table, new_married_table],
                                  ignore_index=True)

    orca.add_table("marriage_table", married_table)

def update_married_households(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals

    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    # Ensure an even number of people get married
    if relevant[relevant["new_mar"] == 1].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 1].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    if relevant[relevant["new_mar"] == 2].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 2].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = (
        relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    )
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")

    # print("Pair people.")
    min_mar = relevant[relevant["new_mar"] == 2]["person_sex"].value_counts().min()
    min_cohab = relevant[relevant["new_mar"] == 1]["person_sex"].value_counts().min()

    female_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")
    ].sample(min_mar)
    male_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")
    ].sample(min_mar)
    female_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")
    ].sample(min_cohab)
    male_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")
    ].sample(min_cohab)

    def brute_force_matching(male, female):
        """Function to run brute force marriage matching.

        TODO: Improve the matchmaking process
        TODO: Account for same-sex marriages

        Args:
            male (DataFrame): DataFrame of the male side
            female (DataFrame): DataFrame of the female side

        Returns:
            DataFrame: DataFrame of newly formed households
        """
        ordered_households = pd.DataFrame()
        male_mar = male.sample(frac=1)
        female_mar = female.sample(frac=1)
        for index in np.arange(female_mar.shape[0]):
            dist = cdist(
                female_mar.iloc[index][["age", "earning"]]
                .to_numpy()
                .reshape((1, 2))
                .astype(float),
                male_mar[["age", "earning"]].to_numpy(),
                "euclidean",
            )
            arg = dist.argmin()
            household_new = pd.DataFrame([female_mar.iloc[index], male_mar.iloc[arg]])
            # household_new["household_group"] = index + 1
            # household_new["new_household_id"] = -99
            # household_new["stay"] = -99
            male_mar = male_mar.drop(male_mar.iloc[arg].name)
            ordered_households = pd.concat([ordered_households, household_new])

        return ordered_households

    cohabitate = brute_force_matching(male_coh, female_coh)
    cohabitate.index.name = "person_id"

    married = brute_force_matching(male_mar, female_mar)
    married.index.name = "person_id"

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result

    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )
    cohabitate = cohabitate.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )

    cohabitate["household_group"] = (
        cohabitate["household_group"] + married["household_group"].max() + 1
    )

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)

    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])]

    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    # married = final[final["new_mar"]==1].copy()
    # cohabitation = final[final["new_mar"]==2].copy()
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[
        final[CONDITION_4]["partner"].values, "new_household_id"
    ] = new_household_ids

    # print("Finished Pairing")
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())

    final["hh_new_id"] = np.where(
        final["stay"].isin([1]),
        final["household_id"],
        np.where(
            final["stay"].isin([0]),
            final["partner_house"],
            final["new_household_id"] + current_max_id + 1,
        ),
    )

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)][
        "household_id"
    ].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]

    households_restructuring = p_df.loc[
        p_df["household_id"].isin(household_ids_reorganized)
    ]

    households_restructuring = households_restructuring.sort_values(
        by=["household_id", "earning"], ascending=False
    )
    households_restructuring.loc[
        households_restructuring.groupby(["household_id"]).head(1).index, "relate"
    ] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        # lcm_county_id=("lcm_county_id", "first"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)

    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    p_df.update(final["MAR"])

    # print("HH SHAPE 2:", p_df["household_id"].unique().shape[0])

    new_hh = household_agg.loc[
        ~household_agg.index.isin(household_df.index.unique())
    ].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = "-1"
    household_df = pd.concat([household_df, new_hh])

    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)

    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        married_table = married_table.append(
            {
                "married": (marriage_list == 1).sum(),
                "cohabitated": (marriage_list == 2).sum(),
            },
            ignore_index=True,
        )
    orca.add_table("marriage_table", married_table)

def update_cohabitating_households(persons, households, cohabitate_list):
    """
    Updating households and persons after cohabitation model.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_local_cols = persons_df.columns
    households_df = orca.get_table("households").local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    households_local_cols = households_df.columns
    married_hh = cohabitate_list.index[cohabitate_list == 2].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].to_list()

    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"] == 13),"relate",] = 1
    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"].isin([1, 0])),"MAR"] = 1

    persons_df = (persons_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id"))

    leaving_person_index = persons_df.index[(persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)]

    leaving_house = persons_df.loc[leaving_person_index].copy()

    leaving_house["relate"] = 0

    persons_df = persons_df.drop(leaving_person_index)

    # Update characteristics for households staying
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    households_new = persons_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_head", "sum"),age_of_head=("age_head", "sum"), workers=("worker", "sum"),hispanic_status_of_head=("hispanic", "sum"),persons=("person", "sum"),children=("child", "sum"),seniors=("senior", "sum"),gt55=("age_gt55", "sum"),
    )

    households_new["hh_age_of_head"] = np.where(households_new["age_of_head"] < 35,"lt35",np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    households_new["hispanic_head"] = np.where( households_new["hispanic_status_of_head"] == 1, "yes", "no")
    households_new["hh_children"] = np.where( households_new["children"] >= 1, "yes", "no")
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(households_new["income"] < 30000,"lt30",np.where(households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_df.update(households_new)

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # Create household characteristics for new households formed
    leaving_house["household_id"] = (
        np.arange(len(breakup_hh))
        + max(max_hh_id, households_df.index.max())
        + 1
    )
    leaving_house["person"] = 1
    leaving_house["is_head"] = np.where(leaving_house["relate"] == 0, 1, 0)
    leaving_house["race_head"] = leaving_house["is_head"] * leaving_house["race_id"]
    leaving_house["age_head"] = leaving_house["is_head"] * leaving_house["age"]
    leaving_house["hispanic_head"] = (
        leaving_house["is_head"] * leaving_house["hispanic"]
    )
    leaving_house["child"] = np.where(leaving_house["relate"].isin([2, 3, 4, 14]), 1, 0)
    leaving_house["senior"] = np.where(leaving_house["age"] >= 65, 1, 0)
    leaving_house["age_gt55"] = np.where(leaving_house["age"] >= 55, 1, 0)

    households_new = leaving_house.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    households_new["hh_cars"] = np.where(
        households_new["cars"] == 0,
        "none",
        np.where(households_new["cars"] == 1, "one", "two or more"),
    )
    households_new["tenure"] = "unknown"
    households_new["recent_mover"] = "unknown"
    households_new["sf_detached"] = "unknown"
    households_new["tenure_mover"] = "unknown"
    households_new["block_id"] = "-1"
    households_new["hh_type"] = "-1"
    households_df = pd.concat([households_df, households_new])

    persons_df = pd.concat([persons_df, leaving_house])

    
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

def update_divorce(divorce_list):
    """
    Updating stats for divorced households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        divorce_list (pd.Series): pandas Series of the divorced households

    Returns:
        None
    """
    # print("Updating household stats...")
    households_local_cols = orca.get_table("households").local.columns

    persons_local_cols = orca.get_table("persons").local.columns

    households_df = orca.get_table("households").local

    persons_df = orca.get_table("persons").local

    households_df.loc[divorce_list.index,"divorced"] = divorce_list

    divorce_households = households_df[households_df["divorced"] == 1].copy()
    DIVORCED_HOUSEHOLDS_ID = divorce_households.index.to_list()

    sizes = persons_df[persons_df["household_id"].isin(divorce_list.index) & (persons_df["relate"].isin([0, 1]))].groupby("household_id").size()


    persons_divorce = persons_df[
        persons_df["household_id"].isin(divorce_households.index)
    ].copy()


    divorced_parents = persons_divorce[
        (persons_divorce["relate"].isin([0, 1])) & (persons_divorce["MAR"] == 1)
    ].copy()

    leaving_house = divorced_parents.groupby("household_id").sample(n=1)

    staying_house = persons_divorce[~(persons_divorce.index.isin(leaving_house.index))].copy()

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # give the people leaving a new household id, update their marriage status, and other variables
    leaving_house["relate"] = 0
    leaving_house["MAR"] = 3
    leaving_house["member_id"] = 1
    leaving_house["household_id"] = (
        np.arange(leaving_house.shape[0]) + max_hh_id + 1
    )

    # modify necessary variables for members staying in household
    staying_house["relate"] = np.where(
        staying_house["relate"].isin([1, 0]), 0, staying_house["relate"]
    )
    staying_house["member_id"] = np.where(
        staying_house["member_id"] != 1,
        staying_house["member_id"] - 1,
        staying_house["relate"],
    )
    staying_house["MAR"] = np.where(
        staying_house["MAR"] == 1, 3, staying_house["MAR"]
    )

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    staying_households = staying_house.copy()
    staying_households["person"] = 1
    staying_households["is_head"] = np.where(staying_households["relate"] == 0, 1, 0)
    staying_households["race_head"] = (
        staying_households["is_head"] * staying_households["race_id"]
    )
    staying_households["age_head"] = (
        staying_households["is_head"] * staying_households["age"]
    )
    staying_households["hispanic_head"] = (
        staying_households["is_head"] * staying_households["hispanic"]
    )
    staying_households["child"] = np.where(
        staying_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    staying_households["senior"] = np.where(staying_households["age"] >= 65, 1, 0)
    staying_households["age_gt55"] = np.where(staying_households["age"] >= 55, 1, 0)

    staying_households = staying_households.sort_values(by=["household_id", "relate"])
    staying_household_agg = staying_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    staying_household_agg["gt55"] = np.where(
        staying_household_agg["persons_age_gt55"] > 0, 1, 0
    )
    staying_household_agg["gt2"] = np.where(staying_household_agg["persons"] > 2, 1, 0)

    staying_household_agg["hh_workers"] = np.where(
        staying_household_agg["workers"] == 0,
        "none",
        np.where(staying_household_agg["workers"] == 1, "one", "two or more"),
    )
    staying_household_agg["hh_age_of_head"] = np.where(
        staying_household_agg["age_of_head"] < 35,
        "lt35",
        np.where(staying_household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    staying_household_agg["hh_race_of_head"] = np.where(
        staying_household_agg["race_of_head"] == 1,
        "white",
        np.where(
            staying_household_agg["race_of_head"] == 2,
            "black",
            np.where(
                staying_household_agg["race_of_head"].isin([6, 7]), "asian", "other"
            ),
        ),
    )
    staying_household_agg["hispanic_head"] = np.where(
        staying_household_agg["hispanic_head"] == 1, "yes", "no"
    )
    staying_household_agg["hh_size"] = np.where(
        staying_household_agg["size"] == 1,
        "one",
        np.where(
            staying_household_agg["size"] == 2,
            "two",
            np.where(staying_household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    staying_household_agg["hh_children"] = np.where(
        staying_household_agg["children"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_seniors"] = np.where(
        staying_household_agg["seniors"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_income"] = np.where(
        staying_household_agg["income"] < 30000,
        "lt30",
        np.where(
            staying_household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                staying_household_agg["income"] < 100,
                "gt60-lt100",
                np.where(staying_household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    staying_household_agg.index.name = "household_id"

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    new_households = leaving_house.copy()
    new_households["person"] = 1
    new_households["is_head"] = np.where(new_households["relate"] == 0, 1, 0)
    new_households["race_head"] = new_households["is_head"] * new_households["race_id"]
    new_households["age_head"] = new_households["is_head"] * new_households["age"]
    new_households["hispanic_head"] = (
        new_households["is_head"] * new_households["hispanic"]
    )
    new_households["child"] = np.where(
        new_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    new_households["senior"] = np.where(new_households["age"] >= 65, 1, 0)
    new_households["age_gt55"] = np.where(new_households["age"] >= 55, 1, 0)

    new_households = new_households.sort_values(by=["household_id", "relate"])
    household_agg = new_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["sf_detached"] = "unknown"
    household_agg["serialno"] = "unknown"
    household_agg["tenure"] = "unknown"
    household_agg["tenure_mover"] = "unknown"
    household_agg["recent_mover"] = "unknown"
    household_agg["cars"] = np.random.choice([0, 1], size=household_agg.shape[0])

    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hispanic_status_of_head"] = np.where(
        household_agg["hispanic_head"] == "yes", 1, 0
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_agg["hh_cars"] = np.where(
        household_agg["cars"] == 0,
        "none",
        np.where(household_agg["cars"] == 1, "one", "two or more"),
    )
    household_agg["block_id"] = "-1"
    household_agg["lcm_county_id"] = "-1"
    household_agg["hh_type"] = 1
    household_agg["household_type"] = 1
    household_agg["serialno"] = "-1"
    household_agg["birth"] = -99
    household_agg["divorced"] = -99

    households_df.update(staying_household_agg)

    hh_ids_p_table = np.hstack((staying_house["household_id"].unique(), leaving_house["household_id"].unique()))
    df_p = persons_df.combine_first(staying_house[persons_local_cols])
    df_p = df_p.combine_first(leaving_house[persons_local_cols])
    hh_ids_hh_table = np.hstack((households_df.index, household_agg.index))

    # merge all in one persons and households table
    new_households = pd.concat([households_df[households_local_cols], household_agg[households_local_cols]])
    persons_df.update(staying_house[persons_local_cols])
    persons_df.update(leaving_house[persons_local_cols])

    orca.add_table("households", new_households[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])

    
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if new_households.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = new_households.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

    # print("Updating divorce metrics...")
    divorce_table = orca.get_table("divorce_table").to_frame()
    if divorce_table.empty:
        divorce_table = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
    else:
        new_divorce = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
        divorce_table = pd.concat([divorce_table, new_divorce], ignore_index=True)
    orca.add_table("divorce_table", divorce_table)
 
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
    marriage_list = simulation_mnl(data, marriage_coeffs)

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
    cohabitate_x_list = simulation_mnl(data, cohabitation_coeffs)
    
    ######### UPDATING
    print("Restructuring households:")
    print("Cohabitations..")
    update_cohabitating_households(persons, households, cohabitate_x_list)
    print_household_stats()
    
    print("Marriages..")
    update_married_households_random(persons, households, marriage_list)
    print_household_stats()
    fix_erroneous_households(persons, households)
    print_household_stats()
    
    print("Divorces..")
    update_divorce(divorce_list)
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

@orca.step("print_household_stats")
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

@orca.step("print_marr_stats")
def print_marr_stats():
    """Genrates marrital status stats
    """
    persons_stats = orca.get_table("persons").local
    persons_stats = persons_stats[persons_stats["age"]>=15]
    print(persons_stats["MAR"].value_counts().sort_values())

@orca.step("laborforce_model")
def laborforce_model(persons, year):
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
    households_df = orca.get_table("households").local
    income_summary = orca.get_table("income_dist").local
    persons_cols = orca.get_injectable("persons_local_cols")
    households_cols = orca.get_injectable("households_local_cols")
    workforce_stats_df = orca.get_table("workforce_stats").to_frame()

    # get observed data
    num_unemployed = persons_df[(persons_df["worker"]==0) & (persons_df["age"]>=18)].shape[0]
    observed_stay_unemployed = orca.get_table("observed_entering_workforce").to_frame()
    entering_workforce_share = observed_stay_unemployed[observed_stay_unemployed["year"]==year]["share"]
    entering_workforce_count = round(entering_workforce_share * num_unemployed)
    observed_exit_workforce = orca.get_table("observed_exiting_workforce").to_frame()
    exiting_workforce_share = observed_exit_workforce[observed_exit_workforce["year"]==year]["share"]
    exiting_workforce_count = (exiting_workforce_share * num_unemployed)

    # entering workforce model
    in_workforce_model = mm.get_step("enter_labor_force")
    predicted_remain_unemployed = calibrate_model(in_workforce_model, entering_workforce_count)

    # Exiting workforce model
    out_workforce_model = mm.get_step("exit_labor_force")
    predicted_exit_workforce = calibrate_model(out_workforce_model, exiting_workforce_count)
    
    # Update labor status
    persons_df = update_labor_status(persons_df, predicted_remain_unemployed, predicted_exit_workforce, income_summary)
    households_df = aggregate_household_labor_variables(persons_df, households_df)
    workforce_stats_df = update_workforce_stats_tables(workforce_stats_df, persons_df, year)

    # Adding updated tables
    orca.add_table("workforce_stats", workforce_stats_df)
    orca.add_table("persons", persons_df[persons_cols])
    orca.add_table("households", households_df[households_cols])

#--------------------------------------------------------------------
# WLCM and SLCM
#--------------------------------------------------------------------
@orca.step("work_location")
def work_location(persons):
    """Runs the work location choice model for workers
    in a region. 

    Args:
        persons (Orca table): persons orca table
    """
    # This workaorund is necesary to make the work
    # location choice run in batches, with improves
    # simulation efficiency.
    model = mm.get_step('wlcm')
    model.run(chooser_batch_size = 100000)
    
    # Update work locations table of individuals #TODO: evaluate whether to keep or remove
    persons_work = orca.get_table("persons").to_frame(columns=["work_block_id"])
    persons_work = persons_work.reset_index()
    orca.add_table('work_locations', persons_work.fillna('-1'))

@orca.step("school_location")
def school_location(persons, households, year):
    """Runs the school location assignment model
    for grade school students

    Args:
        persons (Orca table): Orca table of persons
        households (Orca table): Orca table of households
        year (int): simulation year
    """
    persons_df = orca.get_table("persons").local
    students_df = extract_students(persons_df)
    schools_df = orca.get_table("schools").to_frame()
    blocks_districts = orca.get_table("blocks_districts").to_frame()
    households_df = orca.get_table("households").local
    households_df = households_df.reset_index()
    households_districts = households_df[["household_id", "block_id"]].merge(
        blocks_districts, left_on="block_id", right_on="GEOID10")
    students_df = students_df.merge(
        households_districts.reset_index(), on="household_id")
    students_df = students_df[students_df["STUDENT_SCHOOL_LEVEL"]==students_df["DET_DIST_TYPE"]].copy()
    student_groups = create_student_groups(students_df)

    assigned_students_list = assign_schools(student_groups,
                                            blocks_districts,
                                            schools_df)

    school_assignment_df = create_results_table(students_df, assigned_students_list, year)

    orca.add_table("school_locations", school_assignment_df[["person_id", "school_id"]])

@orca.step("mlcm_postprocessing")
def mlcm_postprocessing(persons):
    """Geographically assign work and school locations
    to workers and students

    Args:
        persons (Orca table): Orca table of persons
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
            persons_local_columns = orca.get_injectable("persons_local_cols")
            updated, updated_persons = deduplicate_updated_households(updated, persons_df, metadata)
            orca.add_table("persons", updated_persons.loc[:,persons_local_columns])

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
            # breakpoint()
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
    return simple_relocation(households, .034, "block_id")


def simple_relocation(choosers, relocation_rate, fieldname):
    print("Total agents: %d" % len(choosers))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))
    print("Assigning for relocation...")
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate * len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname, pd.Series('-1', index=chooser_ids))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))

# -----------------------------------------------------------------------------------------
# POSTPROCESSING
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
            export_demo_table(table) 


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
        add_variables = ["add_temp_variables"]
        start_of_year_models = ["status_report"]
        demo_models = [
            "aging_model",
            # "laborforce_model",
            # "households_reorg",
            # "kids_moving_model",
            "fatality_model",
            "birth_model",
            "education_model",
        ]
        pre_processing_steps = price_models + ["build_networks", "generate_outputs", "update_travel_data"]
        rem_variables = ["remove_temp_variables"]
        export_demo_steps = ["export_demo_stats"]
        household_stats = ["household_stats"]
        school_models = ["school_location"]
        end_of_year_models = ["generate_outputs"]
        work_models = ["work_location"]
        mlcm_postprocessing = ["mlcm_postprocessing"]
        income_model = ["income_model"]
        steps_all_years = (
            start_of_year_models
            + demo_models
            # + work_models
            # + school_models
            + price_models
            + developer_models
            + household_models
            + employment_models
            + end_of_year_models
            # + mlcm_postprocessing
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
