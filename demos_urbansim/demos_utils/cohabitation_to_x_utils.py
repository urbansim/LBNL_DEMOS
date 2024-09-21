import pandas as pd
import numpy as np
from demos_utils.utils import aggregate_household_data

def get_cohabitation_to_x_eligible_households(persons_df):
    """
    Identifies households eligible for the cohabitation_to_x model.

    This function filters the `persons_df` DataFrame to find individuals who are 
    currently in a cohabitation relationship (indicated by `relate` value of 13), 
    are not married (`MAR` not equal to 1), and are at least 15 years old. It 
    returns a list of unique household IDs that meet these criteria, making them 
    eligible for the cohabitation_to_x model.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including relationship status and age.

    Returns:
        np.ndarray: An array of unique household IDs eligible for the cohabitation_to_x model.
    """
    eligible_households = (
        persons_df[(persons_df["relate"] == 13) & \
                (persons_df["MAR"]!=1) & \
                ((persons_df["age"]>=15))]["household_id"].unique().astype(int)
    )
    return eligible_households

def process_cohabitation_to_marriage(persons_df, cohabitation_to_marriage):
    married_condition = persons_df["household_id"].isin(cohabitation_to_marriage)
    persons_df.loc[married_condition & (persons_df["relate"] == 13), "relate"] = 1
    persons_df.loc[married_condition & (persons_df["relate"].isin([1, 0])), "MAR"] = 1
    return persons_df

def process_cohabitation_breakups(persons_df, cohabitate_list, metadata):
    """
    Process households where cohabitation is ending.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        cohabitate_list (pd.Series): Series indicating households undergoing cohabitation changes.

    Returns:
        tuple: Updated persons_df and DataFrame of persons leaving their households.
    """
    # Identify households breaking up
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].tolist()

    # Create a boolean mask for the leaving persons
    leaving_mask = (persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)

    # Extract leaving persons
    leaving_house = persons_df[leaving_mask].copy()
    
    # Update the status of leaving persons
    leaving_house["relate"] = 0

    # Remove leaving persons from the original DataFrame
    persons_df = persons_df[~leaving_mask]

    max_hh_id = max(metadata.loc["max_hh_id", "value"], persons_df["household_id"].max())
    leaving_house["household_id"] = np.arange(len(leaving_house)) + max_hh_id + 1
    return persons_df, leaving_house

def update_cohabitating_households(persons_df, households_df, cohabitate_list, metadata):
    """
    Updates the persons and households dataframes to reflect changes due to cohabitation events.

    This function processes a list of households undergoing cohabitation changes, updating the 
    `persons_df` and `households_df` to reflect changes in household composition and individual 
    statuses. It assigns new household IDs to individuals leaving their current households and 
    updates relationship statuses and other demographic attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including household IDs 
                                   and relationship status.
        households_df (pd.DataFrame): DataFrame containing household-level data, indexed by household IDs.
        cohabitate_list (pd.Series): Series indicating households undergoing cohabitation changes.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum household ID.

    Returns:
        tuple: Updated `persons_df` and `households_df` reflecting the cohabitation events.
    """
    # Incorporate lcm_county_id into persons_df
    persons_df = persons_df.reset_index().set_index("household_id")
    persons_df["lcm_county_id"] = households_df["lcm_county_id"]
    persons_df = persons_df.reset_index().set_index("person_id")

    households_local_cols = households_df.columns
    # Get the list of households undergoing cohabitation to marriage
    cohabitation_to_marriage = cohabitate_list.index[cohabitate_list == 2].to_list()
    persons_df = process_cohabitation_to_marriage(persons_df, cohabitation_to_marriage)
    persons_df, leaving_house = process_cohabitation_breakups(persons_df, cohabitate_list, metadata)

    persons_df, existing_households = aggregate_household_data(persons_df, households_df)
    households_df.update(existing_households)
    leaving_house, new_households = aggregate_household_data(leaving_house, households_df)

    households_df = pd.concat([households_df, new_households])
    persons_df = pd.concat([persons_df, leaving_house])
    
    return persons_df, households_df