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


def update_education_status(persons_df, student_list, year):
    """
    Function to update the student status in persons table based
    on the

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    # Pull Data

    persons_df["stop"] = student_list
    persons_df["stop"].fillna(2, inplace=True)

    dropping_out = persons_df.loc[persons_df["stop"] == 1].copy()
    staying_school = persons_df.loc[persons_df["stop"] == 0].copy()

    dropping_out.loc[:, "student"] = 0
    staying_school.loc[:, "student"] = 1

    # Update education level for individuals staying in school

    persons_df.loc[persons_df["age"] == 3, "edu"] = 2
    persons_df.loc[persons_df["age"].isin([4, 5]), "edu"] = 4

    # high school and high school graduates proportions
    hs_proportions = persons_df[persons_df["edu"].isin([15, 16])]["edu"].value_counts(
        normalize=True
    )
    hs_grad_proportions = persons_df[persons_df["edu"].isin([16, 17])]["edu"].value_counts(
        normalize=True
    )
    # Students all the way to grade 10
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].between(4, 13, inclusive="both"),
        staying_school["edu"] + 1,
        staying_school["edu"],
    )
    # Students in grade 11 move to either 15 or 16
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 14,
        np.random.choice([15, 16], p=[hs_proportions[15], hs_proportions[16]]),
        staying_school["edu"],
    )
    # Students in grade 12 either get hs degree or GED
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 15,
        np.random.choice([16, 17], p=[hs_grad_proportions[16], hs_grad_proportions[17]]),
        staying_school["edu"],
    )
    # Students with GED or HS Degree move to college
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].isin([16, 17]), 18, staying_school["edu"]
    )
    # Students with one year of college move to the next
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 18, 19, staying_school["edu"]
    )

    # Update education levels
    persons_df.update(staying_school)
    persons_df.update(dropping_out)

    return persons_df