import pandas as pd
import orca
import numpy as np

def update_birth_eligibility_count_table(birth_eligible_hh_count_df, eligible_household_ids, year):
    """
    Update the birth eligibility count table with the number of eligible households.

    Args:
        birth_eligible_hh_count_df (pd.DataFrame): Existing DataFrame containing the history of 
                                       birth eligibility counts. Can be empty initially.
        eligible_household_ids (list): List of household IDs that are eligible for births.
        year (int): The current year for which the birth eligibility is being recorded.

    Returns:
        pd.DataFrame: Updated birth eligibility count table with a new row for the current year.
    """
    birth_eligible_hh_count_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [len(eligible_household_ids)]
            })
    if birth_eligible_hh_count_df.empty:
        birth_eligible_hh_count_df = birth_eligible_hh_count_df_new
    else:
        birth_eligible_hh_count_df = pd.concat([birth_eligible_hh_count_df, birth_eligible_hh_count_df_new],
         ignore_index=True)
    return birth_eligible_hh_count_df

def update_births_predictions_table(births_table, year, birth_list):
    """
    Update the births predictions table with the number of predicted births for the current year.

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

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   age, gender, and household IDs.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        list: A list of household IDs that are eligible for the birth model.
    """
    # Specifying the eligibility condition for births
    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
    )

    # Subset of eligible households
    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()
    eligible_household_ids = households_df.loc[ELIGIBLE_HH].index.to_list()
    return eligible_household_ids

def get_highest_person_index(persons_df, metadata):
    """
    Get the highest person index in the persons table.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   person IDs.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum 
                                 person ID.

    Returns:
        int: The highest person index in the persons table.
    """
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
    """
    Get the household IDs of households with predicted births.

    Args:
        birth_list (pd.Series): Series indicating the households with predicted births.

    Returns:
        list: A list of household IDs that are predicted to have a birth.
    """
    return list(birth_list[birth_list == 1].index)

def create_newborns_dataframe(house_indices, highest_index):
    """
    Create a new dataframe for newborns.

    Args:
        house_indices (list): List of household IDs with predicted births.
        highest_index (int): The highest person index in the persons table.

    Returns:
        pd.DataFrame: A new dataframe for newborns with initialized attributes.
    """
    # initialize the newborns dataframe
    newborns = pd.DataFrame(house_indices, columns=["household_id"])
    # assign unique person IDs
    newborns.index += highest_index + 1
    newborns.index.name = "person_id"
    return newborns

def assign_basic_attributes(newborns):
    """
    Assign basic attributes to newborns.

    Args:
        newborns (pd.DataFrame): DataFrame containing newborns with initialized attributes.     

    Returns:
        pd.DataFrame: DataFrame containing newborns with assigned attributes.
    """
    newborns["age"] = 0 # age
    newborns["edu"] = 0 # education
    newborns["earning"] = 0 # earnings
    newborns["hours"] = 0 # hours worked
    newborns["relate"] = 2 # relationship status
    newborns["MAR"] = 5 # marital status
    newborns["sex"] = np.random.choice([1, 2]) # sex
    newborns["student"] = 0 # student status
    newborns["education_group"] = "lte17" # education group
    newborns["age_group"] = "lte20" # age group
    newborns["person_age"] = "19 and under" # person age
    newborns["person_sex"] = newborns["sex"].map({1: "male", 2: "female"}) # person sex
    newborns["child"] = 1 # child status
    newborns["senior"] = 0 # senior status
    newborns["dead"] = -99 # dead status
    newborns["person"] = 1 # person status
    newborns["work_at_home"] = 0 # work at home status
    newborns["worker"] = 0 # worker status
    newborns["work_block_id"] = "-1" # work block id
    newborns["work_zone_id"] = "-1" # work zone id
    newborns["workplace_taz"] = "-1" # workplace taz
    newborns["school_block_id"] = "-1" # school block id
    newborns["school_id"] = "-1" # school id
    newborns["school_taz"] = "-1" # school taz
    newborns["school_zone_id"] = "-1" # school zone id

    return newborns


def assign_race_attributes(newborns, persons_df, households_df):
    """
    Assign race attributes to newborns.

    This function assigns race attributes to newborns based on the race of the household head.

    Args:
        newborns (pd.DataFrame): DataFrame containing newborns with initialized attributes.
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   household IDs and demographic attributes.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        pd.DataFrame: DataFrame containing newborns with assigned race attributes.
    """
    # Get the number of races in each household
    household_races = persons_df.groupby("household_id")
            .agg(num_races=("race_id", "nunique"))
            .reset_index()
            .merge(households_df["race_of_head"].reset_index(), on="household_id")
    # Assign race attributes to newborns
    newborns = newborns.reset_index().merge(household_races, on="household_id")
    newborns["race_id"] = np.where(newborns["num_races"] == 1, newborns["race_of_head"], 9)
    newborns["race"] = newborns["race_id"].map({
        1: "white", 2: "black", 3: "other", 4: "other", 5: "other",
        6: "other", 7: "other", 8: "other", 9: "other",
    })
    return newborns


def assign_hispanic_attributes(newborns, persons_df):
    """
    Assign Hispanic attributes to newborns based on the Hispanic status 
    of the household head.

    Args:
        newborns (pd.DataFrame): DataFrame containing newborns with initialized attributes.
        persons_df (pd.DataFrame): DataFrame containing person-level data

    Returns:
        pd.DataFrame: DataFrame containing newborns with assigned Hispanic attributes.
    """
    # Get the Hispanic status of the household head
    heads = persons_df[persons_df["relate"] == 0]
    heads = heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]]
    newborns = newborns.reset_index()
    # Assign Hispanic attributes to newborns
    return newborns.merge(heads, on="household_id").set_index("person_id")

def assign_member_ids(newborns, persons_df):
    """
    Assign member IDs to newborns.

    Args:
        newborns (pd.DataFrame): DataFrame of newborns.
        persons_df (pd.DataFrame): DataFrame containing person-level data.

    Returns:
        pd.DataFrame: DataFrame of newborns with assigned member IDs.
    """
    grave = orca.get_table("pop_over_time").to_frame()
    all_people = pd.concat([grave, persons_df[["member_id", "household_id"]]]) if not grave.empty else persons_df[["member_id", "household_id"]]
    max_member_id = all_people.groupby("household_id").agg({"member_id": "max"}) + 1
    return (newborns.reset_index()
            .merge(max_member_id, left_on="household_id", right_index=True)
            .set_index("person_id"))

def update_households(households_df, house_indices):
    """
    Update the household characteristics to reflect new births.

    Args:
        households_df (pd.DataFrame): DataFrame containing household-level data.
        house_indices (list): List of household IDs with predicted births.

    Returns:
        pd.DataFrame: household-level data with updated characteristics.
    """
    # index the households with births
    households_babies = households_df.loc[house_indices]
    # update the household characteristics
    households_babies["hh_children"] = "yes"
    households_babies["persons"] += 1
    households_babies["gt2"] = np.where(households_babies["persons"] >= 2, 1, 0)
    households_babies["hh_size"] = np.where(
        households_babies["persons"] == 1, "one",
        np.where(households_babies["persons"] == 2, "two",
                 np.where(households_babies["persons"] == 3, "three", "four or more"))
    )
    # update the households table
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