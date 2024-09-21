import pandas as pd
import numpy as np
from demos_utils.utils import aggregate_household_data

def get_divorce_eligible_household_ids(persons_df):
    """
    Identifies households eligible for divorce processing.

    This function filters the `persons_df` DataFrame to find households where 
    there are exactly two individuals in a married relationship (indicated by 
    `relate` values of 0 or 1 and `MAR` equal to 1). It returns a list of unique 
    household IDs that meet these criteria, making them eligible for divorce processing.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including relationship status and marital status.

    Returns:
        list: A list of unique household IDs eligible for divorce processing.
    """
    eligible_households = list(
        persons_df[(persons_df["relate"].isin([0, 1])) & 
                   (persons_df["MAR"] == 1)]["household_id"].unique().astype(int)
    )
    sizes = (
        persons_df[persons_df["household_id"].isin(eligible_households) & 
                   (persons_df["relate"].isin([0, 1]))]
        .groupby("household_id").size()
    )
    eligible_households = sizes[sizes == 2].index.to_list()
    return eligible_households

def identify_divorce_outcomes(persons_df, divorcing_household_ids, metadata):
    """
    Identifies persons staying in and leaving households undergoing divorce.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        divorcing_household_ids (pd.Index or list): IDs of households undergoing divorce.

    Returns:
        tuple: (staying_persons, leaving_persons)
            - staying_persons (pd.DataFrame): Persons staying in their households after divorce.
            - leaving_persons (pd.DataFrame): Persons leaving their households due to divorce.
    """
    # Get persons in divorcing households
    persons_in_divorcing_households = persons_df[persons_df["household_id"].isin(divorcing_household_ids)]

    # Identify married couples
    married_couples = persons_in_divorcing_households[
        (persons_in_divorcing_households["relate"].isin([0, 1])) & 
        (persons_in_divorcing_households["MAR"] == 1)
    ]

    # Randomly select one person from each couple to leave
    leaving_persons = married_couples.groupby("household_id").sample(n=1)

    # Identify persons staying (including non-couple members)
    staying_persons = persons_in_divorcing_households[
        ~persons_in_divorcing_households.index.isin(leaving_persons.index)
    ]

    max_hh_id = metadata.loc["max_hh_id", "value"]

        # give the people leaving a new household id, update their marriage status, and other variables
    leaving_persons["relate"] = 0
    leaving_persons["MAR"] = 3
    leaving_persons["member_id"] = 1
    
    leaving_persons["household_id"] = (
        np.arange(leaving_persons.shape[0]) + max_hh_id + 1
    )
    staying_persons["relate"] = np.where(
        staying_persons["relate"].isin([1, 0]), 0, staying_persons["relate"]
    )
    staying_persons["member_id"] = np.where(
        staying_persons["member_id"] != 1,
        staying_persons["member_id"] - 1,
        staying_persons["relate"],
    )
    staying_persons["MAR"] = np.where(
        staying_persons["MAR"] == 1, 3, staying_persons["MAR"]
    )

    return staying_persons, leaving_persons

def update_divorce(persons_df, households_df, divorce_list, metadata):
    """
    Updates the persons and households dataframes to reflect divorce events.

    This function processes a list of households undergoing divorce, updating the 
    `persons_df` and `households_df` to reflect changes in household composition 
    and individual statuses. It assigns new household IDs to individuals leaving 
    their current households and updates marital status and other demographic 
    attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including household IDs and marital status.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        divorce_list (pd.Series): Series indicating households undergoing divorce.
        metadata (pd.DataFrame): DataFrame containing metadata, including the 
                                 maximum household ID.

    Returns:
        tuple: Updated `persons_df` and `households_df` reflecting the divorce events.
    """
    households_df.loc[divorce_list.index,"divorced"] = divorce_list
    divorcing_household_ids = households_df[households_df["divorced"] == 1].index.to_list()
    # Identify staying and leaving persons
    staying_persons_df, departing_spouses_df = identify_divorce_outcomes(persons_df, divorcing_household_ids, metadata)
    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    staying_persons_df, staying_households_df = aggregate_household_data(staying_persons_df, households_df)
    departing_spouses_df, new_households_df = aggregate_household_data(departing_spouses_df, households_df, initialize_new_households=True)
    #staying_households_df.index.name = "household_id"
    households_df.update(staying_households_df)
    households_df = pd.concat([households_df, new_households_df])
    persons_df.update(staying_persons_df)
    persons_df.update(departing_spouses_df)

    return persons_df, households_df

def update_divorce_predictions(divorce_table, divorce_list):
    """
    Updates the divorce predictions table with new data.

    This function updates the `divorce_table` DataFrame by adding the count of 
    newly divorced households from the `divorce_list`. If the `divorce_table` is 
    empty, it initializes it with the current counts. Otherwise, it appends the 
    new counts to the existing table.

    Args:
        divorce_table (pd.DataFrame): DataFrame containing historical divorce data.
        divorce_list (pd.Series): Series indicating the divorce status of households.

    Returns:
        pd.DataFrame: Updated DataFrame with the new divorce counts.
    """
    if divorce_table.empty:
        divorce_table = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
    else:
        new_divorce = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
        divorce_table = pd.concat([divorce_table, new_divorce], ignore_index=True)
    return divorce_table

def print_household_stats(persons_df, households_df):
    """
    Prints statistics about households from both the persons and households tables.

    This function calculates and prints the number of unique households and persons 
    from the `persons_df` and `households_df` DataFrames. It also identifies and 
    prints the number of missing households, as well as households with multiple 
    heads or cohabitating individuals.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including household IDs and relationship status.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        None
    """
    # Printing households size from different tables
    print("Households size from persons table:", persons_df["household_id"].nunique())
    print("Households size from households table:", households_df.index.nunique())
    print("Persons Size:", persons_df.index.nunique())

    # Missing households
    missing_households = set(persons_df["household_id"].unique()) - set(households_df.index.unique())
    print("Missing hh:", len(missing_households))

    # Calculating relationships
    persons_df["relate_0"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["relate_1"] = np.where(persons_df["relate"] == 1, 1, 0)
    persons_df["relate_13"] = np.where(persons_df["relate"] == 13, 1, 0)
    
    persons_df_sum = persons_df.groupby("household_id").agg(
        relate_1=("relate_1", "sum"),
        relate_13=("relate_13", "sum"),
        relate_0=("relate_0", "sum")
    )

    # Printing statistics about households
    print("Households with multiple 0:", (persons_df_sum["relate_0"] > 1).sum())
    print("Households with multiple 1:", (persons_df_sum["relate_1"] > 1).sum())
    print("Households with multiple 13:", (persons_df_sum["relate_13"] > 1).sum())
    print("Households with 1 and 13:", ((persons_df_sum["relate_1"] * persons_df_sum["relate_13"]) > 0).sum())