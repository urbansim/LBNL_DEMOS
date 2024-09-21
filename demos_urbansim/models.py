import os
import warnings
from operator import index
warnings.filterwarnings("ignore")

import indicators
import numpy as np
import orca
import pandana as pdna
import pandas as pd
from google.cloud import storage
from scipy.spatial.distance import cdist
from urbansim.developer import developer
from demos_urbansim.demos_utils.education_utils import (
    update_education_status,
    extract_students,
    create_student_groups,
    assign_schools,
    create_results_table,
)
from demos_urbansim.demos_utils.household_reorg_utils import (
    update_divorce,
    update_divorce_predictions,
    print_household_stats,
    get_divorce_eligible_household_ids,
)
from demos_urbansim.demos_utils.single_to_x_utils import (
    update_married_households,
    update_married_predictions,
    get_marriage_eligible_persons,
    update_marrital_status_stats,
)
from demos_urbansim.demos_utils.cohabitation_to_x_utils import (
    process_cohabitation_to_marriage,
    get_cohabitation_to_x_eligible_households,
    update_cohabitating_households,
)
from demos_urbansim.demos_utils.kids_move_utils import (
    update_households_after_kids,
    update_kids_moving_table,
)
from demos_urbansim.demos_utils.simulation_utils import simulation_mnl, calibrate_model
from demos_urbansim.demos_utils.birth_utils import (
    update_birth,
    get_birth_eligible_households,
    update_births_predictions_table,
    update_birth_eligibility_count_table,
)
from demos_urbansim.demos_utils.laborforce_utils import (
    fetch_observed_labor_force_entries_exits,
    update_workforce_stats_tables,
    aggregate_household_labor_variables,
    sample_income,
    update_labor_status,
)
from demos_urbansim.demos_utils.utils import (
    update_metadata,
    export_demo_table,
    deduplicate_updated_households,
    deduplicate_multihead_households,
)
from demos_urbansim.demos_utils.mortality_model_utils import *
from demos_urbansim.demos_utils.income_utils import update_income

# import demo_models
from urbansim.models import GrowthRateTransition, transition
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import BinaryLogitStep, OLSRegressionStep

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


@orca.step("add_temp_variables")
def add_temp_variables():
    """Adds temporary variables to the persons table.

    - 'dead': Initialized to -99, used to track mortality status
    - 'stop': Initialized to -99, purpose to be determined
    - 'kid_moves': Initialized to -99, likely used to track children's relocations

    These variables are temporary and will be removed after processing.
    """
    persons = orca.get_table("persons").local
    persons.loc[:, "dead"] = -99
    persons.loc[:, "stop"] = -99
    persons.loc[:, "kid_moves"] = -99

    orca.add_table("persons", persons)


@orca.step("remove_temp_variables")
def remove_temp_variables():
    """Removes temporary variables from the persons table.

    Temporary variables removed:
    - 'dead': Used to track mortality status
    - 'stop': Purpose to be determined
    - 'kid_moves': Likely used to track children's relocations

    These variables were added temporarily and are no longer needed after processing.
    """
    persons = orca.get_table("persons").local
    persons = persons.drop(columns=["dead", "stop", "kid_moves"])
    orca.add_table("persons", persons)


# -----------------------------------------------
# DEMOS
# -----------------------------------------------
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
    households_df = orca.get_table("households").local
    # FIXME: NEED TO TRACK POP_OVER_TIME
    grave = orca.get_table("pop_over_time").to_frame()
    metadata = orca.get_table("metadata").to_frame()
    graveyard_table = orca.get_table("graveyard").to_frame()
    mortalities_df = orca.get_table("mortalities").to_frame()
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    observed_fatalities_df = orca.get_table("observed_fatalities_data").to_frame()
    target_count = observed_fatalities_df.query(f"year == {year}")["count"].squeeze()

    # Calibrate the mortality model
    mortality = mm.get_step("mortality_model")
    fatality_list = calibrate_model(mortality, target_count)
    predicted_fatalities = fatality_list.sum()
    # Update households and persons tables
    persons_df, households_df, grave_persons = remove_dead_persons(
        persons_df, households_df, fatality_list, year
    )

    metadata = update_metadata(metadata, households_df, persons_df)
    graveyard_table = update_graveyard_table(graveyard_table, grave_persons)
    # Update or create the mortalities table
    mortalities_df = update_mortalities_table(
        mortalities_df, predicted_fatalities, year
    )
    # Add tables
    orca.add_table("mortalities", mortalities_df)
    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("graveyard", graveyard_table[persons_local_columns])
    orca.add_table("households", households_df[households_local_columns])
    orca.add_table("metadata", metadata)


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
    households_df = households.local
    # Increment the age of the persons table
    persons_df["age"] += 1
    # Updating the tables
    persons_df, households_df = aggregate_household_data(persons_df, households_df)

    # Update age in the persons table
    orca.get_table("persons").update_col("age", persons_df["age"])
    # Update age related columns in the households table
    columns_to_update = ["age_of_head", "hh_age_of_head", "hh_children", "gt55", "hh_seniors",]
    for column in columns_to_update:
        orca.get_table("households").update_col(column, households_df[column])


@orca.step("income_model")
def income_model(persons, households, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    # Pulling data
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    income_rates = orca.get_table("income_rates").to_frame()
    # Updating income
    persons_df, households_df = update_income(
        persons_df, households_df, income_rates, year
    )
    # Adding updated tables
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
    edu_model = mm.get_step("education_model")
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
    # Pulling data
    households_df = households.local
    persons_df = persons.local
    households_df["birth"] = -99
    orca.add_table("households", households_df)
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    btable_df = orca.get_table("btable").to_frame()
    birth_eligible_hh_count_df = orca.get_table("birth_eligible_hh_count").to_frame()
    metadata = orca.get_table("metadata").to_frame()
    # Getting eligible households
    eligible_household_ids = get_birth_eligible_households(persons_df, households_df)
    birth_eligible_hh_count_df = update_birth_eligibility_count_table(
        birth_eligible_hh_count_df, eligible_household_ids, year
    )
    # Run model
    birth = mm.get_step("birth_model")
    eligible_household_ids = str(eligible_household_ids)
    birth.filters = "index in " + eligible_household_ids
    birth.out_filters = "index in " + eligible_household_ids
    # Calibrate model
    observed_births = orca.get_table("observed_births_data").to_frame()
    target_count = observed_births[observed_births["year"] == year]["count"]
    birth_list = calibrate_model(birth, target_count)
    print(f"{target_count.sum()} observed births.")
    print(f"{birth_list.sum()} predicted births.")

    # Update persons and households
    persons_df, households_df = update_birth(
        persons_df, households_df, birth_list, metadata
    )
    # Update births predictions table
    btable_df = update_births_predictions_table(btable_df, year, birth_list)
    # Update metadata
    metadata = update_metadata(metadata, households_df, persons_df)
    # Update tables
    orca.add_table("persons", persons_df.loc[:, persons_local_columns])
    orca.add_table("households", households_df.loc[:, households_local_columns])
    orca.add_table("metadata", metadata)
    orca.add_table("birth_eligible_hh_count", birth_eligible_hh_count_df)
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
    households_df = orca.get_table("households").local
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_local_cols = orca.get_injectable("households_local_cols")
    metadata = orca.get_table("metadata").to_frame()
    kids_moving_table = orca.get_table("kids_move_table").to_frame()
    # Running model
    kids_moving_model = mm.get_step("kids_moving_model")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)
    # Post-process persons and households
    # after kids move
    persons_df, households_df = update_households_after_kids(
        persons_df, households_df, kids_moving, metadata
    )
    metadata = update_metadata(metadata, households_df, persons_df)
    kids_moving_table = update_kids_moving_table(kids_moving_table, kids_moving)
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_table("metadata", metadata)
    orca.add_table("kids_move_table", kids_moving_table)


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
    # Pulling data
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    households_local_columns = orca.get_injectable("households_local_cols")
    metadata = orca.get_table("metadata").to_frame()
    marrital = orca.get_table("marrital").to_frame()
    married_table = orca.get_table("marriage_table").to_frame()
    divorce_table = orca.get_table("divorce_table").to_frame()
    # ----------------------------------------
    # Marriage Model
    # ----------------------------------------
    marriage_model = orca.get_injectable("single_to_x_model")
    marriage_coeffs = pd.DataFrame(marriage_model["model_coeffs"])
    marriage_variables = pd.DataFrame(marriage_model["spec_names"])
    marriage_variables = marriage_variables[0].values.tolist()
    marriage_eligible_persons = get_marriage_eligible_persons(persons_df)
    marriage_to_x_data = persons.to_frame(columns=marriage_variables).loc[
        marriage_eligible_persons
    ]
    # Run model
    marriage_list = simulation_mnl(marriage_to_x_data, marriage_coeffs)
    random_match = orca.get_injectable("random_match")
    # ----------------------------------------
    # Divorce Model
    # ----------------------------------------
    households_df["divorced"] = -99
    orca.add_table("households", households_df)
    divorce_eligible_hh_ids = get_divorce_eligible_household_ids(persons_df)
    divorce_model = mm.get_step("divorce_model")
    divorce_eligible_hh_ids = str(divorce_eligible_hh_ids)
    divorce_model.filters = "index in " + divorce_eligible_hh_ids
    divorce_model.out_filters = "index in " + divorce_eligible_hh_ids
    # Run Model
    divorce_model.run()
    divorce_list = divorce_model.choices.astype(int)
    # ----------------------------------------
    # Cohabitation_to_X Model
    # ----------------------------------------
    cohabitation_model = orca.get_injectable("cohabitation_to_x_model")
    cohabitation_coeffs = pd.DataFrame(cohabitation_model["model_coeffs"])
    cohabitation_variables = pd.DataFrame(cohabitation_model["spec_names"])
    cohabitation_variables = cohabitation_variables[0].values.tolist()
    eligible_cohab_to_x_households = get_cohabitation_to_x_eligible_households(
        persons_df
    )
    cohabitate_to_x_data = households.to_frame(columns=cohabitation_variables).loc[
        eligible_cohab_to_x_households
    ]
    # Run Model
    cohabitate_to_x_preds = simulation_mnl(cohabitate_to_x_data, cohabitation_coeffs)
    # ----------------------------------------
    # Postprocessing
    # ----------------------------------------
    print("Restructuring households:")
    # postprocess cohabitation to x
    print("Cohabitation-to-X..")
    persons_df, households_df = update_cohabitating_households(
        persons_df, households_df, cohabitate_to_x_preds, metadata
    )
    # update metadata
    metadata = update_metadata(metadata, households_df, persons_df)
    print_household_stats(persons_df, households_df)
    # postprocess single to x
    print("Single-to-X..")
    # update married households
    update_married_households(persons_df, households_df, marriage_list, metadata)
    # update metadata after married
    metadata = update_metadata(metadata, households_df, persons_df)
    # update married predictions table
    married_table = update_married_predictions(married_table, marriage_list)
    print_household_stats(persons_df, households_df)
    # deduplicate multihead households
    persons_df, households_df = deduplicate_multihead_households(
        persons_df, households_df
    )
    print_household_stats(persons_df, households_df)
    # Postprocess divorce
    print("Divorce..")
    persons_df, households_df = update_divorce(
        persons_df, households_df, divorce_list, metadata
    )
    # update metadata after divorce
    metadata = update_metadata(metadata, households_df, persons_df)
    # update divorce predictions table
    divorce_table = update_divorce_predictions(divorce_table, divorce_list)
    print_household_stats(persons_df, households_df)
    persons_df["member_id"] = (
        persons_df.groupby("household_id")["relate"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    # update marrital status stats
    marrital = update_marrital_status_stats(persons_df, marrital, year)
    # Update orca tables
    orca.add_table("marrital", marrital)
    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("households", households_df[households_local_columns])
    orca.add_table("marriage_table", married_table)
    orca.add_table("divorce_table", divorce_table)


@orca.step("print_marr_stats")
def print_marr_stats(persons):
    """
    Function to print the marrital status distribution
    of the population.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Pulling data
    persons_df = orca.get_table("persons").to_frame(columns=["MAR", "age"])
    # Printing marrital status stats
    persons_stats = persons_df[persons_df["age"] >= 15]
    print(persons_stats["MAR"].value_counts().sort_values())


@orca.step("laborforce_model")
def laborforce_model(persons, households, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Pulling relevant data
    persons_df = orca.get_table("persons").local
    persons_df["stay_out"] = -99
    persons_df["leaving_workforce"] = -99
    orca.add_table("persons", persons_df)
    households_df = orca.get_table("households").local
    income_summary = orca.get_table("income_dist").local
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_local_cols = orca.get_injectable("households_local_cols")
    workforce_stats_df = orca.get_table("workforce_stats").to_frame()
    observed_stay_unemployed = orca.get_table("observed_entering_workforce").to_frame()
    observed_exit_workforce = orca.get_table("observed_exiting_workforce").to_frame()
    # Get observed labor force entries and exits
    exiting_workforce_count, entering_workforce_count = (
        fetch_observed_labor_force_entries_exits(
            persons_df, observed_stay_unemployed, observed_exit_workforce, year
        )
    )
    # Running entering workforce model
    in_workforce_model = mm.get_step("enter_labor_force_model")
    predicted_remain_unemployed = calibrate_model(
        in_workforce_model, entering_workforce_count
    )
    # Running exiting workforce model
    out_workforce_model = mm.get_step("exit_labor_force_model")
    predicted_exit_workforce = calibrate_model(
        out_workforce_model, exiting_workforce_count
    )
    # Update labor status and postprorcess data
    persons_df = update_labor_status(
        persons_df,
        predicted_remain_unemployed,
        predicted_exit_workforce,
        income_summary,
    )
    households_df = aggregate_household_labor_variables(persons_df, households_df)
    workforce_stats_df = update_workforce_stats_tables(
        workforce_stats_df, persons_df, year
    )
    # Adding updated tables
    orca.add_table("workforce_stats", workforce_stats_df)
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_table("households", households_df[households_local_cols])


# --------------------------------------------------------------------
# WLCM and SLCM
# --------------------------------------------------------------------
@orca.step("work_location")
def work_location(persons):
    """Runs the work location choice model for workers
    in a region.

    Args:
        persons (Orca table): persons orca table
    """
    # Pulling data
    persons_work = orca.get_table("persons").to_frame(columns=["work_block_id"])
    # NOTE: This workaround of using chooser_batch_size=100000
    # is essential for executing the work
    # location choice model in batches of 100,000 
    # individuals at a time. This approach significantly
    # enhances simulation efficiency by 
    # reducing memory usage and improving processing speed,
    # especially for large populations.
    model = mm.get_step("wlcm")
    model.run(chooser_batch_size=100000)
    # Update work locations table of individuals
    persons_work = persons_work.reset_index()
    orca.add_table("work_locations", persons_work.fillna("-1"))


@orca.step("school_location")
def school_location(persons, households, year):
    """Runs the school location assignment model
    for grade school students

    Args:
        persons (Orca table): Orca table of persons
        households (Orca table): Orca table of households
        year (int): simulation year
    """
    # Pulling data
    persons_df = orca.get_table("persons").local
    schools_df = orca.get_table("schools").to_frame()
    blocks_districts = orca.get_table("blocks_districts").to_frame()
    households_df = orca.get_table("households").local
    households_df = households_df.reset_index()
    # Pre-processing, matching households to districts
    households_districts = households_df[["household_id", "block_id"]].merge(
        blocks_districts, left_on="block_id", right_on="GEOID10"
    )
    # Extracting students
    students_df = extract_students(persons_df)
    students_df = students_df.merge(
        households_districts.reset_index(), on="household_id"
    )
    # Keeping only students with districts of their levels.
    students_df = students_df[
        students_df["STUDENT_SCHOOL_LEVEL"] == students_df["DET_DIST_TYPE"]
    ].copy()

    student_groups = create_student_groups(students_df)
    assigned_students_list = assign_schools(
        student_groups, blocks_districts, schools_df
    )

    school_assignment_df = create_results_table(
        students_df, assigned_students_list, year
    )
    # Adding tables
    orca.add_table(
        "student_school_assignment", school_assignment_df[["person_id", "school_id"]]
    )


@orca.step("mlcm_postprocessing")
def mlcm_postprocessing(persons):
    """Geographically assign work and school locations
    to workers and students

    Args:
        persons (Orca table): Orca table of persons
    """
    # Pulling data
    persons_df = orca.get_table("persons").local
    persons_local_cols = orca.get_injectable("persons_local_cols")
    geoid_to_zone = orca.get_table("geoid_to_zone").to_frame()
    # Updating work locations (blocks and TAZs)
    persons_df = update_work_locations(persons_df, geoid_to_zone)
    orca.add_table("persons", persons_df[persons_local_cols])


def update_work_locations(persons_df, geoid_to_zone):
    """
    Update work locations for persons based on their work block ID.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person information.
        geoid_to_zone (pd.DataFrame): DataFrame mapping GEOIDs to zones.

    Returns:
        pd.DataFrame: Updated persons DataFrame with work locations.
    """
    # Pre-processing
    persons_df = persons_df.reset_index()
    geoid_to_zone = geoid_to_zone.rename(
        columns={"GEOID10": "work_block_id", "zone_id": "workplace_taz"}
    )

    # Assigning tazs and block ids based on work block id
    persons_df = persons_df.merge(
        geoid_to_zone[["work_block_id", "workplace_taz"]],
        on="work_block_id",
        how="left",
        suffixes=("", "_replace"),
    )

    # Update workplace_taz and work_zone_id
    persons_df["workplace_taz"] = persons_df["workplace_taz_replace"].fillna("-1")
    persons_df["work_zone_id"] = persons_df["workplace_taz"]

    # Clean up and set index
    persons_df = persons_df.drop(columns=["workplace_taz_replace"])
    persons_df = persons_df.set_index("person_id")

    return persons_df


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
    workers_no_location = len(
        persons_df[(persons_df["worker"] == 1) & (persons_df["work_block_id"] == "-1")]
    )
    persons_no_location = len(persons_df[persons_df["work_block_id"] == "-1"])

    # Calculate and print the shares
    share_workers_no_location = (
        workers_no_location / total_workers if total_workers else 0
    )
    share_persons_no_location = (
        persons_no_location / total_persons if total_persons else 0
    )
    print("Share of workers with no work location:", share_workers_no_location)
    print("Share of people with no work location:", share_persons_no_location)


# -----------------------------------------------------------------------------------------
# TRANSITION
# -----------------------------------------------------------------------------------------


@orca.step("household_transition")
def household_transition(households, persons, year, metadata):
    """
    Run the household transition model.
    Transition models represent the in/out-migration of households
    in the region.

    Args:
        households (Orca table): Orca table of households
        persons (Orca table): Orca table of persons
        year (int): simulation year
        metadata (Orca table): Orca table of metadata
    """
    linked_tables = {"persons": (persons, "household_id")}
    if ("annual_household_control_totals" in orca.list_tables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        control_totals = orca.get_table("annual_household_control_totals").to_frame()
        full_transition(
            households,
            control_totals,
            "total",
            year,
            "block_id",
            linked_tables=linked_tables,
        )
    elif ("household_growth_rate" in orca.list_injectables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        rate = orca.get_injectable("household_growth_rate")
        simple_transition(
            households,
            rate,
            "block_id",
            set_year_built=True,
            linked_tables=linked_tables,
        )
    elif "hsize_ct" in orca.list_tables():
        control_totals = orca.get_table("hsize_ct").to_frame()
        full_transition(
            households, control_totals, "total_number_of_households", year, "block_id"
        )
    else:
        control_totals = orca.get_table("hct").to_frame()
        if "hh_type" in control_totals.columns:
            if control_totals[control_totals.index == year].hh_type.min() == -1:
                control_totals = control_totals[["total_number_of_households"]]
        full_transition(
            households,
            control_totals,
            "total_number_of_households",
            year,
            "block_id",
            linked_tables=linked_tables,
        )
    households_df = orca.get_table("households").local
    households_df.loc[households_df["block_id"] == "-1", "lcm_county_id"] = "-1"
    households_df.index.rename("household_id", inplace=True)
    persons_df = orca.get_table("persons").local

    orca.add_table("households", households_df)
    orca.add_table("persons", persons_df)

    metadata_df = orca.get_table("metadata").to_frame()
    max_hh_id = metadata_df.loc["max_hh_id", "value"]
    max_p_id = metadata_df.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata_df.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata_df.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata_df)


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
            tran = transition.TabularTotalsTransition(
                ct_sub, totals_column, accounting_column
            )
            updated_sub, added_sub, copied_sub, removed_sub = tran.transition(
                agnt_sub, year
            )
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
            updated, updated_persons = deduplicate_updated_households(
                updated, persons_df, metadata
            )
            orca.add_table("persons", updated_persons.loc[:, persons_local_columns])

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
        starting_index, starting_index + len(new_rows), dtype=np.int
    )
    new_rows.index.name = table.index.name

    return pd.concat([table, new_rows])


@orca.step("households_relocation_basic")
def households_relocation_basic(households):
    """
    orca step to run a simple relocation model.

    Parameters
    ----------
    households : DataFrameWrapper
        Table of households

    Returns
    -------
    function to run the simple relocation model
    """
    return simple_relocation(households, 0.034, "block_id")


def simple_relocation(choosers, relocation_rate, fieldname):
    """
    Run a simple relocation model.

    Parameters
    ----------
    choosers : DataFrameWrapper
        Table of choosers
    relocation_rate : float
        Rate of relocation
    fieldname : str
        Fieldname of the choosers table to update with the relocation status

    Returns
    -------
    None
    """
    print("Total agents: %d" % len(choosers))
    print(
        "Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0)
    )
    print("Assigning for relocation...")
    chooser_ids = np.random.choice(
        choosers.index, size=int(relocation_rate * len(choosers)), replace=False
    )
    choosers.update_col_from_series(fieldname, pd.Series("-1", index=chooser_ids))
    print(
        "Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0)
    )


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
        for table in [
            "pop_over_time",
            "hh_size_over_time",
            "age_over_time",
            "edu_over_time",
            "income_over_time",
            "kids_move_table",
            "divorce_table",
            "marriage_table",
            "btable",
            "age_dist_over_time",
            "pop_size_over_time",
            "student_population",
            "mortalities",
            "birth_eligible_hh_count",
            "marrital",
        ]:
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
        household_models = (
            ["household_transition"] + ["households_relocation_basic"] + hlcm_models
        )
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
        household_models = (
            ["household_transition"]
            + ["households_relocation_basic"]
            + ["hlcm" + str(segment) for segment in range(1, 11)]
        )
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
            # "aging_model",
            # "laborforce_model",
            "households_reorg",
            # "kids_moving_model",
            "fatality_model",
            "birth_model",
            # "education_model",
        ]
        pre_processing_steps = price_models + [
            "build_networks",
            "generate_outputs",
            "update_travel_data",
        ]
        rem_variables = ["remove_temp_variables"]
        export_demo_steps = ["export_demo_stats"]
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
            # + price_models
            # + developer_models
            # + household_models
            # + employment_models
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
