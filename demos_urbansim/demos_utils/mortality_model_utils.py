import numpy as np
import orca
import pandas as pd
from demos_utils.utils import aggregate_household_data

def remove_deceased_houseolds(persons_df, households_df):
    """
    Remove deceased households from the persons and households tables.
    
    Args:
        persons_df (pd.DataFrame): DataFrame of living persons
        households_df (pd.DataFrame): DataFrame of households
    
    Returns:
        pd.DataFrame: Updated persons DataFrame with deceased households removed
        pd.DataFrame: Updated households DataFrame with deceased households removed
    """
    # Get household mortality stats
    household_mortality_stats_df = persons_df.groupby("household_id").agg(
        household_size=("dead", "size"),
        num_dead=("dead", "sum")
    )
    # Get households where everyone dies
    dead_households = household_mortality_stats_df.query("household_size == num_dead").index.unique()
    # Get persons in those households
    grave_persons = persons_df[persons_df["household_id"].isin(dead_households)].copy()
    # Remove from persons table
    persons_df = persons_df[~persons_df["household_id"].isin(dead_households)]
    # Remove from households table
    households_df = households_df[~households_df.index.isin(dead_households)]
    return persons_df, households_df

def update_marital_status_for_widows(persons_df, dead_df):
    """
    Update marital status for persons whose spouse/partner has died.
    
    Args:
        persons_df (pd.DataFrame): DataFrame of living persons
        dead_df (pd.DataFrame): DataFrame of deceased persons
    
    Returns:
        pd.DataFrame: Updated persons DataFrame
    """
    # Identify households with a dead spouse
    dead_spouse_households = dead_df[dead_df["relate"].isin([0, 1, 13])]["household_id"].values

    # Identify widows (both heads and partners) using boolean indexing
    widow_indices = persons_df[
        (persons_df["household_id"].isin(dead_spouse_households)) &
        (persons_df["relate"].isin([0, 1, 13]))
    ].index
    # Update marital status for widows
    persons_df.loc[widow_indices, "MAR"] = 3
    persons_df["MAR"] = persons_df["MAR"].astype(int)
    return persons_df

def restructure_headless_households(persons_df, dead_df):
    """
    Restructure headless households where the head has died.
    
    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        dead_df (pd.DataFrame): DataFrame containing deceased persons.
    
    Returns:
        pd.DataFrame: Updated persons DataFrame with reorganized households.
    """
    # Get households where the head has died
    dead_heads = dead_df[dead_df["relate"] == 0]
    # Get persons in those households
    persons_df["relate"] = persons_df["relate"].astype(int)
    subset_persons_df = persons_df[persons_df["household_id"].isin(dead_heads["household_id"])]
    # Restructure households
    if len(subset_persons_df) > 0:
        subset_persons_df = subset_persons_df.sort_values("relate")
        subset_persons_df = subset_persons_df[["household_id", "relate", "age"]]
        subset_persons_df = subset_persons_df.groupby("household_id").apply(rez)
        persons_df.loc[subset_persons_df.index, "relate"] = subset_persons_df["relate"]
        persons_df["relate"] = persons_df["relate"].astype(int)
    # generate head and partner flags
    persons_df["is_head"] = (persons_df["relate"] == 0).astype(int)
    persons_df["is_partner"] = (persons_df["relate"] == 1).astype(int)
    # debugging step, counting number of heads and partners
    household_counts = persons_df.groupby("household_id").agg({
        "is_head": "sum",
        "is_partner": "sum"
    })
    # keep only valid households
    valid_households = household_counts[
        (household_counts["is_head"] <= 1) & 
        (household_counts["is_partner"] <= 1)
    ].index

    # Keep only valid households
    persons_df = persons_df[persons_df["household_id"].isin(valid_households)]

    # Clean up temporary columns and ensure 'relate' is integer type
    persons_df = persons_df.drop(columns=["is_head", "is_partner"])
    persons_df["relate"] = persons_df["relate"].astype(int)

    return persons_df

def remove_dead_persons(persons_df, households_df, fatality_list, year):
    """
    This function updates the persons table from the output of the fatality model.
    Takes in the persons and households orca tables.

    Args:
        persons (DataFramWrapper): DataFramWrapper of persons table
        households (DataFramWrapper): DataFramWrapper of households table
        fatality_list (pd.Series): Pandas Series of fatality list
    """
    # initialize dead column
    persons_df["dead"] = -99
    # update dead column with fatality list
    persons_df["dead"] = fatality_list
    graveyard = persons_df[persons_df["dead"] == 1].copy()

    # HOUSEHOLD WHERE EVERYONE DIES
    household_mortality_stats_df = persons_df.groupby("household_id").agg(
        household_size=("dead", "size"),
        num_dead=("dead", "sum")
    )
    dead_households = household_mortality_stats_df.query("household_size == num_dead").index.unique()
    grave_persons = persons_df[persons_df["household_id"].isin(dead_households)].copy()
    
    # Remove from persons table
    persons_df = persons_df[~persons_df["household_id"].isin(dead_households)]
    # Remove from households table
    households_df = households_df[~households_df.index.isin(dead_households)]

    #--------------------------------------------
    #HOUSEHOLDS WHERE PART OF HOUSEHOLD DIES
    #--------------------------------------------
    # dead
    dead_df = persons_df[persons_df["dead"] == 1].copy()
    # alive
    persons_df = persons_df[persons_df["dead"] == 0].copy()

    # Update marital status for widows
    persons_df = update_marital_status_for_widows(persons_df, dead_df)
    # Restructure headless households
    persons_df = restructure_headless_households(persons_df, dead_df)
    # Aggregate household data
    persons_df, updated_households_df = aggregate_household_data(persons_df, households_df, initialize_new_households=False)
    # Update households table
    households_df.update(updated_households_df)
    households_df = households_df.loc[persons_df["household_id"].unique()]

    #FIXME: problems with returning the grave_persons
    return persons_df, households_df, grave_persons

def rez(group):
    """
    Function to reassign household roles after the death of a household head.
    
    This function takes a group of household members and updates their 'relate' 
    values (which represent their roles within the household) when the original 
    household head has died. It handles different scenarios:
    
    1. If there's a spouse (relate=1), they become the new head.
    2. If there's an unmarried partner (relate=13), they become the new head.
    3. Otherwise, the oldest person becomes the new head, 
       and other roles are adjusted accordingly.
    """
    # Update the relate variable for the group
    if group["relate"].iloc[0] == 1:
        group["relate"].iloc[0] = 0
        return group
    if 13 in group["relate"].values:
        group["relate"].replace(13, 0, inplace=True)
        return group

    # oldest person becomes new head of household
    new_head_idx = group["age"].idxmax()
    # Map new household roles
    map_func = produce_map_func(group.loc[new_head_idx, "relate"])
    group.loc[new_head_idx, "relate"] = 0
    group.relate = group.relate.map(map_func)
    return group

def produce_map_func(old_role):
    """
    Function that uses the relationship mapping in the
    provided table and returns a function that maps
    new household roles.
    """
    # old role is the previous number of the person 
    # who has now been promoted to head of the household
    string_old_role = str(old_role)

    def inner(role):
        rel_map = orca.get_table("rel_map").to_frame()
        if role == 0:
            new_role = 0
        else:
            new_role = rel_map.loc[role, string_old_role]
        return new_role

    # Returns function that takes a persons old role and 
    # gives them a new one based on how the household is restructured
    return inner

def update_mortalities_table(mortalities_df, predicted_fatalities, year):
        mortalities_new_row = pd.DataFrame(
            {"year": [year], "count": [predicted_fatalities]}
        )
        mortalities_df = (pd.concat([mortalities_df, mortalities_new_row], ignore_index=True) 
                        if not mortalities_df.empty else mortalities_new_row)
        return mortalities_df

def update_graveyard_table(graveyard_table, grave_persons):
        if graveyard_table.empty:
            graveyard_table = grave_persons.copy()
        else:
            graveyard_table = pd.concat([graveyard_table, grave_persons])
        return graveyard_table