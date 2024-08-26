import numpy as np
import pandas as pd
import orca

def update_metadata(metadata, households_df, persons_df):
    """
    Update metadata with the latest maximum household and person IDs.

    This function checks the current maximum household and person IDs in the 
    `households_df` and `persons_df`, respectively, and updates the `metadata` 
    DataFrame to reflect these values if they exceed the existing maximums.

    Args:
        metadata (pd.DataFrame): DataFrame containing metadata, including the 
                                 maximum household and person IDs.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including household IDs.

    Returns:
        pd.DataFrame: Updated `metadata` DataFrame with the latest maximum IDs.
    """
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    return metadata

def export_demo_table(table_name):
    """
    Export a specified Orca table to a CSV file.

    This function retrieves an Orca table by its name, converts it to a DataFrame,
    and exports it as a CSV file. The CSV file is named using the table name and 
    a region code, and is saved in a specified output folder.

    Args:
        table_name (str): The name of the Orca table to be exported.

    Returns:
        None
    """
    region_code = orca.get_injectable("region_code")
    output_folder = orca.get_injectable("output_folder")
    df = orca.get_table(table_name).to_frame()
    csv_name = table_name + "_" + region_code + ".csv"
    df.to_csv(output_folder+csv_name, index=False)

def deduplicate_updated_households(updated, persons_df, metadata):
    """
    Deduplicate and update household and person IDs after changes.

    This function ensures that updated households and persons have unique IDs,
    especially after modifications that might have introduced duplicates. It 
    assigns new IDs to duplicate entries and updates the metadata accordingly.

    Args:
        updated (pd.DataFrame): DataFrame containing updated household data, 
                                potentially with duplicates.
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including household IDs.
        metadata (pd.DataFrame): DataFrame containing metadata, including the 
                                 maximum household and person IDs.

    Returns:
        tuple: Updated `updated` DataFrame with unique household IDs and 
               `persons_df` with unique person IDs.
    """
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    unique_hh_ids = updated["household_id"].unique()
    persons_old = persons_df[persons_df["household_id"].isin(unique_hh_ids)]
    updated = updated.sort_values(["household_id"])
    updated["cum_count"] = updated.groupby("household_id").cumcount()
    updated = updated.sort_values(by=["cum_count"], ascending=False)
    updated.loc[:,"new_household_id"] = np.arange(updated.shape[0]) + max_hh_id + 1
    updated.loc[:,"new_household_id"] = np.where(updated["cum_count"]>0, updated["new_household_id"], updated["household_id"])
    sampled_persons = updated.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
    sampled_persons = sampled_persons.sort_values(by=["cum_count"], ascending=False)
    sampled_persons.loc[:,"new_person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
    sampled_persons.loc[:,"person_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_person_id"], sampled_persons["person_id"])
    sampled_persons.loc[:,"household_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_household_id"], sampled_persons["household_id"])
    updated.loc[:,"household_id"] = updated.loc[:, "new_household_id"]
    sampled_persons.set_index("person_id", inplace=True, drop=True)
    updated.set_index("household_id", inplace=True, drop=True)
    return updated, sampled_persons

def aggregate_household_data(persons_df, households_df, initialize_new_households=False):
    """
    Aggregate household data from person-level data.

    This function aggregates person-level data to create household-level summaries.
    It calculates various household attributes such as income, race, age of head,
    number of workers, and more. Optionally, it can initialize new households with
    random attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   household IDs and demographic attributes.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        initialize_new_households (bool, optional): If True, initializes new households 
                                                    with attributes. Defaults to False.

    Returns:
        tuple: Updated `persons_df` and a DataFrame of aggregated household data.
    """
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 7, 9, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)
        
    persons_df = persons_df.sort_values("relate")

    agg_dict = {
        "income": ("earning", "sum"),
        "race_of_head": ("race_head", "sum"),
        "age_of_head": ("age_head", "sum"),
        "workers": ("worker", "sum"),
        "hispanic_status_of_head": ("hispanic_head", "sum"),
        "seniors": ("senior", "sum"),
        "lcm_county_id": ("lcm_county_id", "first"),
        "persons": ("person", "sum"),
        "age_gt55": ("age_gt55", "sum"),
        "children": ("child", "sum"),
    }
    agg_dict = {k: v for k, v in agg_dict.items() if v[0] in persons_df.columns}

    if initialize_new_households:
        persons_df["cars"] = np.random.choice([0, 1, 2], size=len(persons_df))
        agg_dict["cars"] = ("cars", "sum")

    agg_households = persons_df.groupby("household_id").agg(**agg_dict)

    agg_households["hh_age_of_head"] = np.where(
        agg_households["age_of_head"] < 35,
        "lt35",
        np.where(agg_households["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    agg_households["hh_race_of_head"] = np.where(
        agg_households["race_of_head"] == 1,
        "white",
        np.where(
            agg_households["race_of_head"] == 2,
            "black",
            np.where(agg_households["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    agg_households["hispanic_head"] = np.where(
        agg_households["hispanic_status_of_head"] == 1, "yes", "no"
    )
    agg_households["hh_size"] = np.where(
        agg_households["persons"] == 1,
        "one",
        np.where(
            agg_households["persons"] == 2,
            "two",
            np.where(agg_households["persons"] == 3, "three", "four or more"),
        ),
    )
    agg_households["hh_children"] = np.where(
        agg_households["children"] >= 1, "yes", "no"
    )
    agg_households["hh_income"] = np.where(
        agg_households["income"] < 30000,
        "lt30",
        np.where(
            agg_households["income"] < 60,
            "gt30-lt60",
            np.where(
                agg_households["income"] < 100,
                "gt60-lt100",
                np.where(agg_households["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    agg_households["hh_workers"] = np.where(
        agg_households["workers"] == 0,
        "none",
        np.where(agg_households["workers"] == 1, "one", "two or more"),
    )
    agg_households["hh_seniors"] = np.where(agg_households["seniors"] >= 1, "yes", "no")
    agg_households["gt55"] = np.where(agg_households["age_gt55"] > 0, 1, 0)
    agg_households["gt2"] = np.where(agg_households["persons"] > 2, 1, 0)

    #TODO: WHICH ONES SHOULD BE INITIALIZED BY -1?
    if initialize_new_households:
        agg_households["hh_cars"] = np.where(
            agg_households["cars"] == 0,
            "none",
            np.where(agg_households["cars"] == 1, "one", "two or more"),
        )
        agg_households["serialno"] = "-1"
        agg_households["tenure"] = np.random.choice(
            households_df["tenure"].unique(), size=agg_households.shape[0]
        )
        agg_households["recent_mover"] = np.random.choice(
            households_df["recent_mover"].unique(), size=agg_households.shape[0]
        )
        agg_households["sf_detached"] = np.random.choice(
            households_df["sf_detached"].unique(), size=agg_households.shape[0]
        )
        agg_households["tenure_mover"] = np.random.choice(
            households_df["tenure_mover"].unique(), size=agg_households.shape[0]
        )
        agg_households["block_id"] = np.random.choice(
            households_df["block_id"].unique(), size=agg_households.shape[0]
        )
        agg_households["hh_type"] = 0

    return persons_df, agg_households

def deduplicate_multihead_households(persons_df, households_df):
    """
    This function identifies and removes households that contain only non-head 
    members (e.g., spouses or partners without a head) from the `households_df` 
    and `persons_df`. It ensures that each household has a valid head member.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including household IDs and relationship status.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        tuple: Updated `persons_df` and `households_df` with erroneous households removed.
    """
    households_to_drop = persons_df[persons_df['relate'].isin([1, 13])].groupby('household_id')['relate'].nunique().reset_index()
    households_to_drop = households_to_drop[households_to_drop["relate"]==2]["household_id"].to_list()
    households_df = households_df.drop(households_to_drop)
    persons_df = persons_df[~persons_df["household_id"].isin(households_to_drop)]

    return persons_df, households_df