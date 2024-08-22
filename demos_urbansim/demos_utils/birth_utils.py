import pandas as pd
import orca
import numpy as np

def update_birth_eligibility_count_table(btable_elig_df, eligible_household_ids, year):
    """
    Update the birth eligibility count table with the number of eligible households.

    This function adds a new entry to the birth eligibility count table, recording
    the number of households eligible for births in the given year. If the table is
    initially empty, it creates a new DataFrame with the current count.

    Args:
        btable_elig_df (pd.DataFrame): Existing DataFrame containing the history of 
                                       birth eligibility counts. Can be empty initially.
        eligible_household_ids (list): List of household IDs that are eligible for births.
        year (int): The current year for which the birth eligibility is being recorded.

    Returns:
        pd.DataFrame: Updated birth eligibility count table with a new row for the current year.
    """
    btable_elig_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [len(eligible_household_ids)]
            })
    if btable_elig_df.empty:
        btable_elig_df = btable_elig_df_new
    else:
        btable_elig_df = pd.concat([btable_elig_df, btable_elig_df_new],
         ignore_index=True)
    return btable_elig_df

def update_births_predictions_table(births_table, year, birth_list):
    """
    Update the births predictions table with the number of predicted births for the current year.

    This function adds a new entry to the births predictions table, recording the number of 
    predicted births in the given year. If the table is initially empty, it creates a new 
    DataFrame with the current count.

    Args:
        births_table (pd.DataFrame): Existing DataFrame containing the history of predicted 
                                     births. Can be empty initially.
        year (int): The current year for which the birth predictions are being recorded.
        birth_list (pd.Series): Series indicating the households with predicted births.

    Returns:
        pd.DataFrame: Updated births predictions table with a new row for the current year.
    """
    # print("Updating birth metrics...")
    btable_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })
    if births_table.empty:
        births_table = btable_df_new
    else:
        births_table = pd.concat([births_table, btable_df_new], ignore_index=True)
    return births_table

def get_birth_eligible_households(persons_df, households_df):
    """
    Identify households eligible for births based on person-level data.

    This function determines which households are eligible for births by identifying
    individuals within a specified age range and gender. It returns a list of unique
    household IDs that meet the eligibility criteria.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   age, gender, and household IDs.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        list: A list of household IDs that are eligible for the birth model.
    """
    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
    )

    # Subset of eligible households
    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()
    eligible_household_ids = households_df.loc[ELIGIBLE_HH].index.to_list()
    return eligible_household_ids

def get_highest_person_index(persons_df, metadata):
    highest_index = persons_df.index.max()
    max_p_id = metadata.loc["max_p_id", "value"]
    highest_index = max(max_p_id, highest_index)
    grave = orca.get_table("pop_over_time").to_frame()
    if not grave.empty:
        graveyard = orca.get_table("graveyard")
        dead_df = graveyard.to_frame(columns=["member_id", "household_id"])
        highest_dead_index = dead_df.index.max()
        highest_index = max(highest_dead_index, highest_index)
    
    return highest_index
def get_birth_households(birth_list):
    return list(birth_list[birth_list == 1].index)

def create_newborns_dataframe(house_indices, highest_index):
    newborns = pd.DataFrame(house_indices, columns=["household_id"])
    newborns.index += highest_index + 1
    newborns.index.name = "person_id"
    return newborns

def assign_basic_attributes(newborns):
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
        return newborns


def assign_race_attributes(newborns, persons_df, households_df):
    household_races = get_household_races(persons_df, households_df)
    newborns = newborns.reset_index().merge(household_races, on="household_id")
    newborns["race_id"] = np.where(newborns["num_races"] == 1, newborns["race_of_head"], 9)
    newborns["race"] = newborns["race_id"].map({
        1: "white", 2: "black", 3: "other", 4: "other", 5: "other",
        6: "other", 7: "other", 8: "other", 9: "other",
    })
    return newborns

def get_household_races(persons_df, households_df):
    return (persons_df.groupby("household_id")
            .agg(num_races=("race_id", "nunique"))
            .reset_index()
            .merge(households_df["race_of_head"].reset_index(), on="household_id"))

def assign_hispanic_attributes(newborns, persons_df):
    heads = persons_df[persons_df["relate"] == 0]
    heads = heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]]
    newborns = newborns.reset_index()
    return newborns.merge(heads, on="household_id").set_index("person_id")

def assign_member_ids(newborns, persons_df):
    grave = orca.get_table("pop_over_time").to_frame()
    all_people = pd.concat([grave, persons_df[["member_id", "household_id"]]]) if not grave.empty else persons_df[["member_id", "household_id"]]
    max_member_id = all_people.groupby("household_id").agg({"member_id": "max"}) + 1
    return (newborns.reset_index()
            .merge(max_member_id, left_on="household_id", right_index=True)
            .set_index("person_id"))

def update_households(households_df, house_indices):
    households_babies = households_df.loc[house_indices]
    households_babies["hh_children"] = "yes"
    households_babies["persons"] += 1
    households_babies["gt2"] = np.where(households_babies["persons"] >= 2, 1, 0)
    households_babies["hh_size"] = np.where(
        households_babies["persons"] == 1, "one",
        np.where(households_babies["persons"] == 2, "two",
                 np.where(households_babies["persons"] == 3, "three", "four or more"))
    )
    households_df.update(households_babies[households_df.columns])
    return households_df
            
def update_birth(persons_df, households_df, birth_list, metadata):
    """
    Update the persons and households dataframes to reflect new births.

    This function adds new individuals (babies) to the `persons_df` based on the 
    `birth_list`, which indicates households with predicted births. It assigns new 
    person IDs to the babies and updates the `households_df` to reflect the increased 
    household size and other related attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   household IDs and demographic attributes.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        birth_list (pd.Series): Series indicating the households with predicted births.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum 
                                 person ID.

    Returns:
        tuple: Updated `persons_df` with new individuals added and `households_df` 
               reflecting the changes due to new births.
    """
    highest_index = get_highest_person_index(persons_df, metadata)
    house_indices = get_birth_households(birth_list)
    # Initialize babies variables in the persons table.
    newborns = create_newborns_dataframe(house_indices, highest_index)
    newborns = assign_basic_attributes(newborns)
    newborns = assign_race_attributes(newborns, persons_df, households_df)
    newborns = assign_hispanic_attributes(newborns, persons_df)
    newborns = assign_member_ids(newborns, persons_df)
    # Update tables
    households_df = update_households(households_df, house_indices)
    persons_df = pd.concat([persons_df, newborns])
    return persons_df, households_df