import numpy as np
import pandas as pd
from demos_utils.utils import aggregate_household_data

def update_kids_moving_table(kids_moving_table, kids_moving):
    """
    Update the kids_moving_table with the number of kids moving out in the current iteration.

    This function takes the existing kids_moving_table and adds a new row with the count
    of kids moving out in the current iteration. If the kids_moving_table is empty, it
    creates a new DataFrame with the current count.

    Args:
        kids_moving_table (pd.DataFrame): Existing DataFrame containing the history of
                                          kids moving out. Can be empty for the first iteration.
        kids_moving (pd.Series): A boolean Series indicating which kids are moving out
                                 in the current iteration.

    Returns:
        pd.DataFrame: Updated kids_moving_table with a new row added for the current
                      iteration's count of kids moving out.
    """
    new_kids_moving_table = pd.DataFrame(
            {"kids_moving_out": [kids_moving.sum()]}
        )
    if kids_moving_table.empty:
        kids_moving_table = new_kids_moving_table
    else:
        kids_moving_table = pd.concat([kids_moving_table, 
                                    new_kids_moving_table],
                                    ignore_index=True)
    return kids_moving_table


def update_households_after_kids(persons_df, households_df, kids_moving, metadata):
    """
    Add and update households after kids move out.

    This function updates the `persons_df` and `households_df` to reflect changes 
    when children move out of their current households. It assigns new household 
    IDs to individuals moving out and updates household compositions accordingly.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   household IDs and demographic attributes.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        kids_moving (pd.Series): Pandas Series indicating which children are moving 
                                 out of their current households.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum 
                                 household ID.

    Returns:
        tuple: Updated `persons_df` and `households_df` reflecting the changes due 
               to children moving out.
    """
    # print("Updating households...")
    persons_df["moveoutkid"] = kids_moving
    persons_df = persons_df.reset_index()
    # Add county information to persons_df
    persons_df = persons_df.merge(
        households_df[["lcm_county_id"]],
        left_on="household_id",
        right_index=True,
        how="left"
    )
    persons_df = persons_df.set_index("person_id")

    # Get max household_id across simulations
    highest_index = households_df.index.max()
    max_hh_id = metadata.loc["max_hh_id", "value"]    
    current_max_household_id = max(max_hh_id, highest_index)

    # kids_leaving_mask = persons_df["moveoutkid"] == 1
    # households_with_kids_leaving = persons_df.loc[kids_leaving_mask, "household_id"].unique()

    # Aggregate household size and number of kids moving
    household_moving_stats_df = persons_df.groupby("household_id").agg(
        household_size=("moveoutkid", "size"),
        num_moving=("moveoutkid", "sum")
    )

    # Identify households where all members are marked as moving
    nonmoving_households = household_moving_stats_df.query("household_size == num_moving").index.unique()

    # Prevent these households from moving out
    persons_df["moveoutkid"] = np.where(
        persons_df["household_id"].isin(nonmoving_households),
        0,
        persons_df["moveoutkid"])
    # Assign new household IDs to persons who are actually moving out
    persons_df.loc[persons_df["moveoutkid"] == 1, "household_id"] = (
        np.arange(persons_df["moveoutkid"].sum()) + current_max_household_id + 1
    )

    #new_hh
    moving_persons_df = persons_df.loc[persons_df["moveoutkid"] == 1].copy()

    persons_df = persons_df.drop(persons_df[persons_df["moveoutkid"] == 1].index)

    persons_df, old_agg_household = aggregate_household_data(persons_df, households_df)

    households_df.update(old_agg_household)

    moving_persons_df, agg_households = aggregate_household_data(moving_persons_df,
    households_df,
    initialize_new_households=True)

    households_df["birth"] = -99
    households_df["divorced"] = -99

    agg_households["birth"] = -99
    agg_households["divorced"] = -99

    households_df = pd.concat([households_df, agg_households])
    persons_df = pd.concat([persons_df, moving_persons_df])
    
    return persons_df, households_df