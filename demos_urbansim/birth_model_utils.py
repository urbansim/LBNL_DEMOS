import orca
import numpy as np
import pandas as pd
import utils

def update_birth(persons, households, birth_list):
    """
    Update the persons tables with newborns and household sizes

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the persons table
        birth_list (pd.Series): Pandas Series of the households with newborns

    Returns:
        None
    """
    persons_df = persons.local
    households_df = households.local
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_local_cols = orca.get_injectable("households_local_cols")
    metadata = orca.get_table("metadata").to_frame()
    
    newborns = initialize_newborns(persons_df, households_df, birth_list, metadata)
    persons_df = pd.concat([persons_df, newborns])

    households_df = update_households_after_birth(households_df, birth_list)
    
    metadata = utils.update_metadata(metadata, persons_df, households_df)

    orca.add_table("persons", persons_df.loc[:, persons_local_cols])
    orca.add_table("households", households_df.loc[:, households_local_cols])
    orca.add_table("metadata", metadata)

def get_eligible_households(persons_df):
    """Retrieves the list of eligible households
    for the birth model.

    Args:
        persons_df (pd.DataFrame): Table of individual agents

    Returns:
        str: string representation of eligible households, to be
        used with UrbanSim Templates
    """

    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
    )

    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()

    return str(list(ELIGIBLE_HH))

def update_households_after_birth(households_df, birth_list):
    """
    Update households data frame based on the birth list.

    Args:
    households_df (DataFrame): The DataFrame containing household data.
    birth_list (Series): A Series indicating households with new births (1 for birth, 0 otherwise).

    Returns:
    DataFrame: The updated households DataFrame.
    """
    # Find indices of households with new births
    house_indices = birth_list[birth_list == 1].index

    # Update households with new births
    households_df.loc[house_indices, 'hh_children'] = 'yes'
    households_df.loc[house_indices, 'persons'] += 1

    households_df["gt2"] = np.where(households_df['persons'] >= 2, 1, 0)

    households_df["hh_size"] = np.where(households_df["persons"] == 1, "one",
                               np.where(households_df["persons"] == 2, "two",
                               np.where(households_df["persons"] == 3, "three", "four or more")))

    return households_df

def initialize_newborns(persons_df, households_df, birth_list, metadata):
    # Get indices of households with babies
    house_indices = list(birth_list[birth_list == 1].index)

    # Initialize babies variables in the persons table.
    ## TODO: PREFER TO PUT THIS IN A FUNCTION
    newborns = pd.DataFrame(house_indices, columns=["household_id"])
    newborns["age"] = 0
    newborns["edu"] = 0
    newborns["earning"] = 0
    newborns["hours"] = 0
    newborns["relate"] = 2
    newborns["MAR"] = 5
    newborns["sex"] = np.random.choice([1, 2])
    newborns["student"] = 0

    newborns["person_age"] = "19 and under"
    newborns["person_sex"] = newborns["sex"].map({1: "male", 2: "female"})
    newborns["child"] = 1
    newborns["senior"] = 0
    newborns["dead"] = -99
    newborns["person"] = 1
    newborns["work_at_home"] = 0
    newborns["worker"] = 0
    newborns["work_block_id"] = "-1"
    newborns["work_zone_id"] = "-1"
    newborns["workplace_taz"] = "-1"
    newborns["school_block_id"] = "-1"
    newborns["school_id"] = "-1"
    newborns["school_taz"] = "-1"
    newborns["school_zone_id"] = "-1"
    newborns["education_group"] = "lte17"
    newborns["age_group"] = "lte20"

    household_races = (
        persons_df.groupby("household_id")
        .agg(num_races=("race_id", "nunique"))
        .reset_index()
        .merge(households_df["race_of_head"].reset_index(), on="household_id")
    )
    newborns = newborns.reset_index().merge(household_races, on="household_id")
    newborns["race_id"] = np.where(newborns["num_races"] == 1, newborns["race_of_head"], 9)
    newborns["race"] = newborns["race_id"].map(
        {
            1: "white",
            2: "black",
            3: "other",
            4: "other",
            5: "other",
            6: "other",
            7: "other",
            8: "other",
            9: "other",
        }
    )

    # Get heads of households
    heads = persons_df[persons_df["relate"] == 0]
    newborns = (
        newborns.reset_index()
        .merge(
            heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]],
            on="household_id",
        )
    )

    max_p_id = metadata.loc["max_p_id", "value"]

    highest_index = max(max_p_id, persons_df.index.max())

    max_member_id = persons_df.groupby("household_id").agg({"member_id": "max"})
    max_member_id += 1
    max_member_id = max_member_id.reset_index()

    newborns.index += highest_index + 1
    newborns.index.name = "person_id"

    newborns = newborns.reset_index()
    newborns = newborns.merge(max_member_id, on="household_id")
    newborns = newborns.set_index("person_id")

    return newborns

def update_births_table(birth_list, year):
    """Update the births stats table
    after the model run.

    Args:
        birth_list (list): List of birth outcomes
        year (int): Simulation year
    """
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
