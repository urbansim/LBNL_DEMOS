import numpy as np
import pandas as pd

def increment_ages(persons_df):
    """
    Function to increment the age of the persons table and
    update the age of the household head in the household table.

    Args:
        persons_df (pd.DataFrame): DataFrame of the persons table

    Returns:
        persons_df (pd.DataFrame): DataFrame of the persons table with the age incremented
        households_ages (pd.DataFrame): DataFrame of the updated households
        table with only age related columns
    """
    # Increment the age of the persons table
    persons_df["age"] += 1
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    
    # Generate age related columns in the households table
    households_ages = persons_df.groupby(["household_id"]).agg(
        children=("child", "sum"),
        seniors=("senior", "sum"),
        size=("person", "sum"),
        age_of_head=("age_head", "sum"),
    )

    # Generate age related columns in the households table
    households_ages["hh_children"] = np.where(
        households_ages["children"] > 0, "yes", "no"
    )
    households_ages["gt55"] = np.where(households_ages["seniors"] > 0, 1, 0)
    households_ages["hh_seniors"] = np.where(
        households_ages["seniors"] > 0, "yes", "no"
    )
    households_ages["hh_age_of_head"] = np.where(
        households_ages["age_of_head"] < 35,
        "lt35",
        np.where(households_ages["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    
    return persons_df, households_ages