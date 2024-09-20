import pandas as pd
import numpy as np
from demos_utils.utils import aggregate_household_data

def get_marriage_eligible_persons(persons_df):
    """
    Identifies individuals eligible for marriage.

    This function filters the `persons_df` DataFrame to find individuals who are 
    single (not married, `MAR` not equal to 1), at least 15 years old, and not 
    currently cohabitating (not in a household with `relate` values of 0 or 13). 
    It returns the indices of individuals who meet these criteria.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including relationship status and age.

    Returns:
        pd.Index: An index of individuals eligible for marriage.
    """
    cohab_household_ids = persons_df[persons_df["relate"] == 13]["household_id"].unique()

    # Condition for single status
    single_condition = (persons_df["MAR"] != 1) & (persons_df["age"] >= 15)

    # Condition for non-cohabitation
    non_cohab_condition = ~((persons_df["household_id"].isin(cohab_household_ids)) & 
                            ((persons_df["relate"] == 0) | (persons_df["relate"] == 13)))

    # Get indices of eligible people
    eligible_persons = persons_df[single_condition & non_cohab_condition].index
    return eligible_persons

def check_minimum_marriages(marriage_list, min_count=10):
    """Check if there are enough marriages and cohabitations."""
    return ((marriage_list == 1).sum() <= min_count) or ((marriage_list == 2).sum() <= min_count)

def swap(arr):
    result = np.empty_like(arr)
    result[::2] = arr[1::2]
    result[1::2] = arr[::2]
    return result

def first(size):
    result = np.zeros(size)
    result[::2] = result[::2] + 1
    return result

def assign_household_role(size, marriage=True):
    result = np.zeros(size)
    if marriage:
        result[1::2] = result[1::2] + 1
    else:
        result[1::2] = result[1::2] + 13
    return result

def calculate_marriage_counts(candidates, marriage_list):
    """Calculate the number of marriages and cohabitations for each gender."""
    # Combine candidates and marriage_list
    combined = pd.DataFrame({
        'person_sex': candidates['person_sex'],
        'marriage_status': marriage_list
    })
    
    # Count occurrences for each combination of marriage status and gender
    counts = combined.groupby(['marriage_status', 'person_sex']).size().unstack(fill_value=0)
    
    return {
        "marriages": min(counts.loc[2, "female"], counts.loc[2, "male"]),
        "cohabitations": min(counts.loc[1, "female"], counts.loc[1, "male"])
    }

def pair_individuals(group1, group2):
    """Pair individuals from two groups based on age similarity."""
    # Ensure both groups have the same size
    min_size = min(len(group1), len(group2))
    group1 = group1.sort_values("age").iloc[:min_size]
    group2 = group2.sort_values("age").iloc[:min_size]
    
    # Assign pair numbers
    group1["pair_number"] = np.arange(min_size)
    group2["pair_number"] = np.arange(min_size)
    
    paired = pd.concat([group1, group2])
    return paired.sort_values("pair_number")

def process_couples(couples, start_household_id):
    """Process couples for marriage or cohabitation."""
    couples["first"] = first(len(couples))
    couples["partner"] = swap(couples.index)
    couples["partner_house"] = swap(couples["household_id"])
    couples["partner_relate"] = swap(couples["relate"])
    couples["new_household_id"] = -99
    couples["stay"] = -99
    couples = couples[couples["household_id"] != couples["partner_house"]].copy()

    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 3 - this person leaves their household and creates a new household with partner
    conditions = [
        ((couples["first"] == 1) & (couples["relate"] == 0) & (couples["partner_relate"] == 0)),
        ((couples["first"] == 1) & (couples["relate"] == 0) & (couples["partner_relate"] != 0)),
        ((couples["first"] == 1) & (couples["relate"] != 0) & (couples["partner_relate"] == 0)),
        ((couples["first"] == 1) & (couples["relate"] != 0) & (couples["partner_relate"] != 0))
    ]
    choices = [1, 1, 0, 3]
    couples.loc[couples.index, "stay"] = np.select(conditions, choices, default=-99)
    couples.loc[couples["partner"], "stay"] = np.select(conditions, [0, 0, 1, 3], default=-99)

    new_household_mask = couples["stay"] == 3
    new_household_count = new_household_mask.sum()
    new_household_ids = np.arange(start_household_id, start_household_id + new_household_count)
    # breakpoint()
    # Create a Series to map pair numbers to new household IDs
    pair_to_household = pd.Series(new_household_ids, index=couples[new_household_mask]["pair_number"].unique())

    # Assign new household IDs based on pair numbers
    couples.loc[new_household_mask, "new_household_id"] = couples.loc[new_household_mask, "pair_number"].map(pair_to_household)

    couples["household_id"] = np.where(couples["stay"] == 1, couples["household_id"],
                              np.where(couples["stay"] == 0, couples["partner_house"],
                              couples["new_household_id"]))

    return couples

def find_households_needing_reorganization(persons_df, old_household_ids, new_household_ids):
    """
    Identify households that need reorganization due to the head leaving.
    
    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        old_household_ids (pd.Series): Series of original household IDs for individuals who moved.
        new_household_ids (pd.Series): Series of new household IDs for individuals who moved.
    
    Returns:
        np.array: Array of household IDs that need reorganization.
    """
    # Identify households where someone left
    households_with_leavers = old_household_ids[old_household_ids != new_household_ids].unique()
    
    # Check if the leaver was the head of household
    heads_left = persons_df[
        (persons_df["household_id"].isin(households_with_leavers)) & 
        (persons_df["relate"] == 0) & 
        (~persons_df.index.isin(old_household_ids.index))
    ]["household_id"].unique()
    
    return heads_left

def reorganize_households(persons_df, household_ids):
    """
    Reorganize households where the head has left.
    
    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        households_needing_reorganization (np.array): Array of household IDs needing reorganization.
    
    Returns:
        pd.DataFrame: Updated persons DataFrame with reorganized households.
    """
    if len(household_ids) > 0:
        # Select households that need reorganization
        subset_persons_df = persons_df[persons_df["household_id"].isin(household_ids)].reset_index()
        # Sort by household_id and earning, then assign new head of household
        subset_persons_df = subset_persons_df.sort_values(by=["household_id", "earning"], ascending=[True, False])
        # breakpoint()
        person_ids = subset_persons_df.groupby("household_id").first()["person_id"].values
        subset_persons_df = subset_persons_df.set_index("person_id")
        subset_persons_df.loc[person_ids, "relate"] = 0
        
        # Update persons_df with restructured households
        persons_df.update(subset_persons_df)
    
    return persons_df

def get_marriage_candidates(persons_df, marriage_list):
    """
    Get marriage candidates based on the marriage list.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data.
        marriage_list (pd.Series): Series indicating marriage status (1 for cohabitation, 2 for marriage).

    Returns:
        dict: Dictionary containing DataFrames for each category of marriage candidates.
    """
    # Filter candidates using marriage_list indices
    candidates = persons_df.loc[marriage_list.index]
    
    # Calculate minimum numbers of cohab and marriages
    marriage_counts = calculate_marriage_counts(candidates, marriage_list)
    
    if marriage_counts["marriages"] == 0 and marriage_counts["cohabitations"] == 0:
        return {}

    result = {}

    # Select candidates for marriage
    marriage_candidates = candidates[marriage_list == 2]
    result['female_mar'] = marriage_candidates[marriage_candidates['person_sex'] == 'female'].sample(marriage_counts['marriages'])
    result['male_mar'] = marriage_candidates[marriage_candidates['person_sex'] == 'male'].sample(marriage_counts['marriages'])

    # Select candidates for cohabitation
    cohab_candidates = candidates[marriage_list == 1]
    result['female_coh'] = cohab_candidates[cohab_candidates['person_sex'] == 'female'].sample(marriage_counts['cohabitations'])
    result['male_coh'] = cohab_candidates[cohab_candidates['person_sex'] == 'male'].sample(marriage_counts['cohabitations'])

    return result

def update_married_households(persons_df, households_df, marriage_list, metadata, random_match=True):
    """
    Update the marriage status of individuals and create new households.

    This function processes a list of individuals who are getting married or cohabitating,
    updating the `persons_df` and `households_df` to reflect changes in household composition
    and individual statuses. It assigns new household IDs to individuals leaving their current
    households and updates relationship statuses and other demographic attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including household IDs 
                                   and relationship status.
        households_df (pd.DataFrame): DataFrame containing household-level data, indexed by household IDs.
        marriage_list (pd.Series): Series indicating individuals undergoing marriage or cohabitation changes.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum household ID.

    Returns:
        tuple: Updated `persons_df` and `households_df` reflecting the marriage and cohabitation events.
    """
    # if there are less than 10 marriages or cohabitations,
    # return the original data
    if check_minimum_marriages(marriage_list):
        return persons_df, households_df
    # Add household county id to persons df
    persons_df = persons_df.reset_index()
    persons_df = persons_df.merge(households_df[["lcm_county_id"]],
             on=["household_id"])
    persons_df = persons_df.set_index("person_id")

    # In the main function, replace the relevant code with:
    marriage_candidates = get_marriage_candidates(persons_df, marriage_list)

    if not marriage_candidates:
        return persons_df, households_df

    # Continue with the rest of the function using the dictionary
    female_mar = marriage_candidates['female_mar']
    male_mar = marriage_candidates['male_mar']
    female_coh = marriage_candidates['female_coh']
    male_coh = marriage_candidates['male_coh']
    if random_match:
        married = pair_individuals(female_mar, male_mar)
        cohabitate = pair_individuals(female_coh, male_coh)
    else:
        married = brute_force_matching(male_mar, female_mar)
        cohabitate = brute_force_matching(male_coh, female_coh)
    # Assign household groups
    married["household_group"] = married["pair_number"]
    cohabitate["household_group"] = cohabitate["pair_number"] + married["household_group"].max() + 1

    # Sort within household groups by earning
    married = married.sort_values(by=["household_group", "earning"], ascending=[True, False])
    cohabitate = cohabitate.sort_values(by=["household_group", "earning"], ascending=[True, False])
    
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, households_df.index.max())
    married = process_couples(married, start_household_id=current_max_id + 1)
    current_max_id = max(current_max_id, married["household_id"].max())
    cohabitate = process_couples(cohabitate, start_household_id=current_max_id + 1)
    # Assign new household role status
    married["MAR"] = 1
    married["relate"] = assign_household_role(len(married))
    cohabitate["relate"] = assign_household_role(len(cohabitate), False)
    final = pd.concat([married, cohabitate])

    # Store original household IDs before updates
    original_household_ids = persons_df.loc[final.index, "household_id"].copy()

    # Update the household id and relationship status for the individuals
    persons_df.loc[final.index, "household_id"] = final["household_id"]
    persons_df.loc[final.index, "relate"] = final["relate"]
    
    # Update MAR status only for married couples
    married_mask = final["MAR"] == 1
    persons_df.loc[final[married_mask].index, "MAR"] = 1

    # Find households needing reorganization
    households_needing_reorg = find_households_needing_reorganization(
        persons_df, original_household_ids, final["household_id"]
    )

    # Reorganize households where the head has left
    persons_df = reorganize_households(persons_df, households_needing_reorg)
    # Filter households from the persons df
    households_df = households_df.loc[households_df.index.isin(persons_df["household_id"])]

    persons_df = persons_df.sort_values("relate")
    persons_df, household_agg = aggregate_household_data(persons_df, households_df)
    households_df.update(household_agg)

    new_households_df = household_agg.loc[~household_agg.index.isin(households_df.index.unique())].copy()
    new_households_df["serialno"] = "-1"
    new_households_df["cars"] = np.random.choice([0, 1, 2], size=new_households_df.shape[0])
    new_households_df["hispanic_status_of_head"] = "-1"
    new_households_df["tenure"] = "-1"
    new_households_df["recent_mover"] = "-1"
    new_households_df["sf_detached"] = "-1"
    new_households_df["hh_cars"] = np.where(
        new_households_df["cars"] == 0, "none", np.where(new_households_df["cars"] == 1, "one", "two or more")
    )
    new_households_df["tenure_mover"] = "-1"
    new_households_df["block_id"] = "-1"
    new_households_df["hh_type"] = "-1"

    households_df = pd.concat([households_df, new_households_df])

    return persons_df, households_df

def update_married_predictions(married_table, marriage_list):
    """
    Updates the marriage predictions table with new data.

    This function updates the `married_table` DataFrame by adding the count of 
    newly married and cohabitating individuals from the `marriage_list`. If the 
    `married_table` is empty, it initializes it with the current counts. Otherwise, 
    it appends the new counts to the existing table.

    Args:
        married_table (pd.DataFrame): DataFrame containing historical marriage 
                                      and cohabitation data.
        marriage_list (pd.Series): Series indicating the marriage status of individuals, 
                                   where 1 represents married and 2 represents cohabitated.

    Returns:
        pd.DataFrame: Updated DataFrame with the new marriage and cohabitation counts.
    """
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        new_married_table = pd.DataFrame(
                [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
                columns=["married", "cohabitated"]
            )
        married_table = pd.concat([married_table, new_married_table],
                                  ignore_index=True)
    return married_table

def update_marrital_status_stats(persons_df, marrital, year):
    """
    Updates the marital status statistics with new data.

    This function updates the `marital` DataFrame by adding the count of individuals 
    aged 15 and older for each marital status category from the `persons_df`. If the 
    `marital` DataFrame is empty, it initializes it with the current year's data. 
    Otherwise, it appends the new data to the existing DataFrame.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including age and marital status.
        marital (pd.DataFrame): DataFrame containing historical marital status data.
        year (int): The current year for which the data is being updated.

    Returns:
        pd.DataFrame: Updated DataFrame with the new marital status counts.
    """
    if marrital.empty:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        marrital = pd.DataFrame(persons_stats)
        marrital["year"] = year
    else:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        new_marrital = pd.DataFrame(persons_stats)
        new_marrital["year"] = year
        marrital = pd.concat([marrital, new_marrital])
    return marrital

def brute_force_matching(group1, group2):
    """Function to run brute force marriage matching.

    Args:
        male (DataFrame): DataFrame of the male side
        female (DataFrame): DataFrame of the female side

    Returns:
        DataFrame: DataFrame of newly formed households
    """
    ordered_households = pd.DataFrame()
    group1_mar = group1.sample(frac=1)
    group2_mar = group2.sample(frac=1)
    for index in np.arange(group2_mar.shape[0]):
        dist = cdist(
            group2_mar.iloc[index][["age", "earning"]]
            .to_numpy()
            .reshape((1, 2))
            .astype(float),
            group1_mar[["age", "earning"]].to_numpy(),
            "euclidean",
        )
        arg = dist.argmin()
        household_new = pd.DataFrame([group2_mar.iloc[index], group1_mar.iloc[arg]])
        male_mar = male_mar.drop(male_mar.iloc[arg].name)
        ordered_households = pd.concat([ordered_households, household_new])
    ordered_households.index.name = "person_id"
    return ordered_households
