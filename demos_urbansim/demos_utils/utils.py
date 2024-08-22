import numpy as np
import pandas as pd
import orca
from scipy.special import softmax

def simulation_mnl(data, coeffs):
    """
    Simulate choices using a Multinomial Logit (MNL) model.

    This function calculates the utility of each choice alternative using the 
    provided data and coefficients, applies the softmax function to compute 
    choice probabilities, and simulates choices based on these probabilities.

    Args:
        data (pd.DataFrame or np.ndarray): The input data containing features 
                                           for each choice alternative.
        coeffs (np.ndarray): Coefficients for the MNL model, used to calculate 
                             the utility of each choice alternative.

    Returns:
        pd.Series: A Pandas Series containing the simulated choices, indexed 
                   by the input data's index.
    """
    utils = np.dot(data, coeffs)
    base_util = np.zeros(utils.shape[0])
    utils = np.column_stack((base_util, utils))
    probabilities = softmax(utils, axis=1)
    s = probabilities.cumsum(axis=1)
    r = np.random.rand(probabilities.shape[0]).reshape((-1, 1))
    choices = (s < r).sum(axis=1)
    return pd.Series(index=data.index, data=choices)


def increment_ages(persons_df):
    """
    Increment the ages of individuals and update related demographic attributes.

    This function increases the age of each individual in the `persons_df` by one year.
    It also updates various demographic attributes such as child status, head of household
    status, and senior status. Additionally, it generates age-related columns in the 
    households table, summarizing the number of children, seniors, and the age of the 
    household head.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   age, relationship status, and household IDs.

    Returns:
        tuple: Updated `persons_df` with incremented ages and a DataFrame of aggregated 
               age-related data for households.
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


# -------------------------------
# BIRTH TABLE
# -------------------------------
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
    # Pull max person index from persons table
    highest_index = persons_df.index.max()

    # Check if the pop_over_time is an empty dataframe
    grave = orca.get_table("pop_over_time").to_frame()

    # If not empty, update the highest index with max index of all people
    max_p_id = metadata.loc["max_p_id", "value"]

    highest_index = max(max_p_id, highest_index)

    if not grave.empty:
        graveyard = orca.get_table("graveyard")
        dead_df = graveyard.to_frame(columns=["member_id", "household_id"])
        highest_dead_index = dead_df.index.max()
        highest_index = max(highest_dead_index, highest_index)

    # Get heads of households
    heads = persons_df[persons_df["relate"] == 0]

    # Get indices of households with babies
    house_indices = list(birth_list[birth_list == 1].index)

    # Initialize babies variables in the persons table.
    babies = pd.DataFrame(house_indices, columns=["household_id"])
    babies.index += highest_index + 1
    babies.index.name = "person_id"
    babies["age"] = 0
    babies["edu"] = 0
    babies["earning"] = 0
    babies["hours"] = 0
    babies["relate"] = 2
    babies["MAR"] = 5
    babies["sex"] = np.random.choice([1, 2])
    babies["student"] = 0

    babies["person_age"] = "19 and under"
    babies["person_sex"] = babies["sex"].map({1: "male", 2: "female"})
    babies["child"] = 1
    babies["senior"] = 0
    babies["dead"] = -99
    babies["person"] = 1
    babies["work_at_home"] = 0
    babies["worker"] = 0
    babies["work_block_id"] = "-1"
    babies["work_zone_id"] = "-1"
    babies["workplace_taz"] = "-1"
    babies["school_block_id"] = "-1"
    babies["school_id"] = "-1"
    babies["school_taz"] = "-1"
    babies["school_zone_id"] = "-1"
    babies["education_group"] = "lte17"
    babies["age_group"] = "lte20"
    household_races = (
        persons_df.groupby("household_id")
        .agg(num_races=("race_id", "nunique"))
        .reset_index()
        .merge(households_df["race_of_head"].reset_index(), on="household_id")
    )
    babies = babies.reset_index().merge(household_races, on="household_id")
    babies["race_id"] = np.where(babies["num_races"] == 1, babies["race_of_head"], 9)
    babies["race"] = babies["race_id"].map(
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
    babies = (
        babies.reset_index()
        .merge(
            heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]],
            on="household_id",
        )
        .set_index("person_id")
    )

    # Add counter for member_id to not overlap from dead people for households
    if not grave.empty:
        all_people = pd.concat([grave, persons_df[["member_id", "household_id"]]])
    else:
        all_people = persons_df[["member_id", "household_id"]]
    max_member_id = all_people.groupby("household_id").agg({"member_id": "max"})
    max_member_id += 1
    babies = (
        babies.reset_index()
        .merge(max_member_id, left_on="household_id", right_index=True)
        .set_index("person_id")
    )
    households_babies = households_df.loc[house_indices]
    households_babies["hh_children"] = "yes"
    households_babies["persons"] += 1
    households_babies["gt2"] = np.where(households_babies["persons"] >= 2, 1, 0)
    households_babies["hh_size"] = np.where(
        households_babies["persons"] == 1,
        "one",
        np.where(
            households_babies["persons"] == 2,
            "two",
            np.where(households_babies["persons"] == 3, "three", "four or more"),
        ),
    )

    # Update the households table
    households_df.update(households_babies[households_df.columns])
    # Contactenate the final result
    combined_result = pd.concat([persons_df, babies])
    return combined_result, households_df

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

def update_income(persons_df, households_df, income_rates, year):
    """
    Update the income of individuals and households based on income rates for a given year.

    This function adjusts the earnings of individuals in the `persons_df` by applying 
    income growth rates specific to their county and the current year. It then aggregates 
    the updated earnings to compute the total household income and updates the `households_df`.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   earnings and household IDs.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.
        income_rates (pd.DataFrame): DataFrame containing income growth rates by county 
                                     and year.
        year (int): The current year for which the income updates are being applied.

    Returns:
        tuple: Updated `persons_df` and `households_df` with adjusted incomes.
    """
    year_income_rate = income_rates[income_rates["year"] == year]
    hh_counties = households_df["lcm_county_id"].copy()
    persons_df = (persons_df.reset_index().merge(hh_counties.reset_index(), on=["household_id"]).set_index("person_id"))
    persons_df = (persons_df.reset_index().merge(year_income_rate, on=["lcm_county_id"]).set_index("person_id"))
    persons_df["earning"] = persons_df["earning"] * (1 + persons_df["rate"])

    new_incomes = persons_df.groupby("household_id").agg(income=("earning", "sum"))

    households_df.update(new_incomes)
    households_df["income"] = households_df["income"].astype(int)
    return persons_df, households_df

def update_workforce_stats_tables(workforce_df, persons_df, year):
    """
    Update workforce statistics tables with current year data.

    This function updates the workforce statistics by adding new data for the 
    current year, including the number of individuals entering and exiting the 
    workforce. It appends this data to the existing workforce statistics DataFrame.

    Args:
        workforce_df (pd.DataFrame): DataFrame containing historical workforce statistics.
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   employment status.
        year (int): The current year for which the statistics are being updated.

    Returns:
        pd.DataFrame: Updated `workforce_df` with new statistics for the current year.
    """
    new_workforce_df = pd.DataFrame(
            data={"year": [year], "entering_workforce": [persons_df[persons_df["remain_unemployed"]==0].shape[0]],
            "exiting_workforce": [persons_df[persons_df["exit_workforce"]==1].shape[0]]})
    if workforce_df.empty:
        workforce_df = new_workforce_df
    else:
        workforce_df = pd.concat([workforce_df, new_workforce_df])
    return workforce_df

def aggregate_household_labor_variables(persons_df, households_df):
    """
    Aggregate labor-related variables at the household level.

    This function calculates household-level labor statistics by aggregating 
    person-level data. It computes the total number of workers and the total 
    income for each household, and categorizes households based on the number 
    of workers.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including worker status and earnings.
        households_df (pd.DataFrame): DataFrame containing household-level data, 
                                      indexed by household IDs.

    Returns:
        pd.DataFrame: Updated `households_df` with aggregated labor variables.
    """
    # TODO: Similarly, do something for work from home
    household_incomes = persons_df.groupby("household_id").agg(
        sum_workers = ("worker", "sum"),
        income = ("earning", "sum")
    )
    
    household_incomes["hh_workers"] = np.where(
        household_incomes["sum_workers"] == 0,
        "none",
        np.where(household_incomes["sum_workers"] == 1, "one", "two or more"))
          
    households_df.update(household_incomes)

    return households_df

def sample_income(mean, std):
    """
    Generate a sample income using a log-normal distribution.

    This function generates a random income value based on a log-normal distribution
    defined by the specified mean and standard deviation.

    Args:
        mean (float): The mean of the log-normal distribution.
        std (float): The standard deviation of the log-normal distribution.

    Returns:
        float: A randomly sampled income value.
    """
    return np.random.lognormal(mean, std)

def update_labor_status(persons_df, stay_unemployed_list, exit_workforce_list, income_summary):
    """
    Update the labor status and income of individuals in the persons DataFrame.

    This function updates the labor status of individuals based on whether they remain 
    unemployed or exit the workforce. It also updates their income based on age and 
    education group, using a provided income summary.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   age, education, and current labor status.
        stay_unemployed_list (pd.Series): Series indicating individuals who will remain 
                                          unemployed.
        exit_workforce_list (pd.Series): Series indicating individuals who will exit the 
                                         workforce.
        income_summary (pd.DataFrame): DataFrame containing income distribution parameters 
                                       (mean and standard deviation) for different age and 
                                       education groups.

    Returns:
        pd.DataFrame: Updated persons_df with modified labor status and income.
    """
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    education_intervals = [0, 18, 22, 200]
    # Define the labels for age and education groups
    age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
    education_labels = ['lte17', '18-21', 'gte22']
    # Create age and education groups with labels
    persons_df['age_group'] = pd.cut(persons_df['age'], bins=age_intervals, labels=age_labels, include_lowest=True)
    persons_df['education_group'] = pd.cut(persons_df['edu'], bins=education_intervals, labels=education_labels, include_lowest=True)
    # Sample income for each individual based on their age and education group
    persons_df = persons_df.reset_index().merge(income_summary, on=['age_group', 'education_group'], how='left').set_index("person_id")
    persons_df['new_earning'] = persons_df.apply(lambda row: sample_income(row['mu'], row['sigma']), axis=1)

    persons_df["exit_workforce"] = exit_workforce_list
    persons_df["exit_workforce"].fillna(2, inplace=True)

    persons_df["remain_unemployed"] = stay_unemployed_list
    persons_df["remain_unemployed"].fillna(2, inplace=True)

    # Update worker status
    persons_df["worker"] = np.where(persons_df["exit_workforce"]==1, 0, persons_df["worker"])
    persons_df["worker"] = np.where(persons_df["remain_unemployed"]==0, 1, persons_df["worker"])
    persons_df["work_at_home"] = persons_df["work_at_home"].fillna(0)

    persons_df.loc[persons_df["exit_workforce"]==1, "earning"] = 0
    persons_df["earning"] = np.where(persons_df["remain_unemployed"]==0, persons_df["new_earning"], persons_df["earning"])

    return persons_df

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
    csv_name = table_name + "_" + region_code +".csv"
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


def calibrate_model(model, target_count, threshold=0.05):
    """
    Calibrate a model to match a target count.

    This function adjusts the model's parameters to ensure that the predicted 
    outcomes closely match the target count. It iteratively updates the model's 
    parameters until the error between the predicted and target counts is within 
    a specified threshold.

    Args:
        model: The model object to be calibrated. It must have `run` and 
               `fitted_parameters` attributes.
        target_count (int or float): The target count that the model's predictions 
                                     should match.
        threshold (float, optional): The acceptable error threshold for calibration. 
                                     Defaults to 0.05.

    Returns:
        np.ndarray: The calibrated predictions from the model.
    """
    model.run()
    predictions = model.choices.astype(int)
    predicted_share = predictions.sum() / predictions.shape[0]
    target_share = target_count / predictions.shape[0]

    error = (predictions.sum() - target_count.sum())/target_count.sum()
    while np.abs(error) >= threshold:
        model.fitted_parameters[0] += np.log(target_count.sum()/predictions.sum())
        model.run()
        predictions = model.choices.astype(int)
        predicted_share = predictions.sum() / predictions.shape[0]
        error = (predictions.sum() - target_count.sum())/target_count.sum()
    return predictions

#------------------------------------------
def fix_erroneous_households(persons_df, households_df):
    """
    Fix households with erroneous compositions.

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


def update_married_households_random(persons_df, households_df, marriage_list, metadata):
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
    # print("Updating persons and households...")
    # household_cols = households.local_columns
    # household_df = households.local
    # persons_cols = persons.local_columns
    # persons_local_cols = persons.local_columns
    hh_df = households_df[["lcm_county_id"]].copy()
    hh_df = hh_df.reset_index()
    persons_df["new_mar"] = marriage_list
    persons_df["new_mar"].fillna(0, inplace=True)
    relevant = persons_df[persons_df["new_mar"] > 0].copy()

    if ((relevant["new_mar"] ==1).sum() <= 10) or ((relevant["new_mar"] ==2).sum() <= 10):
        return None

    relevant.sort_values("new_mar", inplace=True)

    relevant = relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    persons_df = persons_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")


    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result
    
    min_mar_male = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].shape[0]
    min_mar_female = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].shape[0]

    min_cohab_male = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].shape[0]
    min_cohab_female = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].shape[0]
    
    min_mar = int(min(min_mar_male, min_mar_female))
    min_cohab = int(min(min_cohab_male, min_cohab_female))

    if (min_mar == 0) or (min_mar == 0):
        return None

    female_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].sample(min_mar)
    male_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].sample(min_mar)
    female_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].sample(min_cohab)
    male_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].sample(min_cohab)
    

    female_mar = female_mar.sort_values("age")
    male_mar = male_mar.sort_values("age")
    female_coh = female_coh.sort_values("age")
    male_coh = male_coh.sort_values("age")
    female_mar["number"] = np.arange(female_mar.shape[0])
    male_mar["number"] = np.arange(male_mar.shape[0])
    female_coh["number"] = np.arange(female_coh.shape[0])
    male_coh["number"] = np.arange(male_coh.shape[0])
    married = pd.concat([male_mar, female_mar])
    cohabitate = pd.concat([male_coh, female_coh])
    married = married.sort_values(by=["number"])
    cohabitate = cohabitate.sort_values(by=["number"])

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(by=["household_group", "earning"], ascending=[True, False])
    cohabitate = cohabitate.sort_values(by=["household_group", "earning"], ascending=[True, False])

    cohabitate["household_group"] = (cohabitate["household_group"] + married["household_group"].max() + 1)

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)
    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])].copy()

    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[final[CONDITION_4]["partner"].values, "new_household_id"] = new_household_ids

    # print('Finished Pairing')
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, households_df.index.max())
    final["hh_new_id"] = np.where(final["stay"].isin([1]), final["household_id"], np.where(final["stay"].isin([0]),final["partner_house"],final["new_household_id"] + current_max_id + 1))

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)]["household_id"].unique()

    persons_df.loc[final.index, "household_id"] = final["hh_new_id"]
    persons_df.loc[final.index, "relate"] = final["new_relate"]

    households_restructuring = persons_df.loc[persons_df["household_id"].isin(household_ids_reorganized)]

    households_restructuring = households_restructuring.sort_values(by=["household_id", "earning"], ascending=False)
    households_restructuring.loc[households_restructuring.groupby(["household_id"]).head(1).index, "relate"] = 0

    households_df = households_df.loc[households_df.index.isin(persons_df["household_id"])]

    persons_df = persons_df.sort_values("relate")

    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    persons_df = persons_df.sort_values(by=["household_id", "relate"])
    household_agg = persons_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_id", "first"),age_of_head=("age", "first"),size=("person", "sum"),workers=("worker", "sum"),hispanic_head=("hispanic_head", "sum"),persons_age_gt55=("age_gt55", "sum"),seniors=("senior", "sum"),children=("child", "sum"),persons=("person", "sum"),)

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["hh_workers"] = np.where(household_agg["workers"] == 0,"none",np.where(household_agg["workers"] == 1, "one", "two or more"),)
    household_agg["hh_age_of_head"] = np.where(household_agg["age_of_head"] < 35,"lt35",np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    households_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    persons_df["NEW_MAR"] = final["MAR"]
    persons_df["MAR"] = np.where(persons_df["NEW_MAR"].isna(),persons_df["MAR"], persons_df["NEW_MAR"])

    new_hh = household_agg.loc[~household_agg.index.isin(households_df.index.unique())].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = "-1"
    households_df = pd.concat([households_df, new_hh])

    return persons_df, households_df

def update_married_households(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals

    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    # Ensure an even number of people get married
    if relevant[relevant["new_mar"] == 1].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 1].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    if relevant[relevant["new_mar"] == 2].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 2].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = (
        relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    )
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")

    # print("Pair people.")
    min_mar = relevant[relevant["new_mar"] == 2]["person_sex"].value_counts().min()
    min_cohab = relevant[relevant["new_mar"] == 1]["person_sex"].value_counts().min()

    female_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")
    ].sample(min_mar)
    male_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")
    ].sample(min_mar)
    female_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")
    ].sample(min_cohab)
    male_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")
    ].sample(min_cohab)

    def brute_force_matching(male, female):
        """Function to run brute force marriage matching.

        TODO: Improve the matchmaking process
        TODO: Account for same-sex marriages

        Args:
            male (DataFrame): DataFrame of the male side
            female (DataFrame): DataFrame of the female side

        Returns:
            DataFrame: DataFrame of newly formed households
        """
        ordered_households = pd.DataFrame()
        male_mar = male.sample(frac=1)
        female_mar = female.sample(frac=1)
        for index in np.arange(female_mar.shape[0]):
            dist = cdist(
                female_mar.iloc[index][["age", "earning"]]
                .to_numpy()
                .reshape((1, 2))
                .astype(float),
                male_mar[["age", "earning"]].to_numpy(),
                "euclidean",
            )
            arg = dist.argmin()
            household_new = pd.DataFrame([female_mar.iloc[index], male_mar.iloc[arg]])
            # household_new["household_group"] = index + 1
            # household_new["new_household_id"] = -99
            # household_new["stay"] = -99
            male_mar = male_mar.drop(male_mar.iloc[arg].name)
            ordered_households = pd.concat([ordered_households, household_new])

        return ordered_households

    cohabitate = brute_force_matching(male_coh, female_coh)
    cohabitate.index.name = "person_id"

    married = brute_force_matching(male_mar, female_mar)
    married.index.name = "person_id"

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result

    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )
    cohabitate = cohabitate.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )

    cohabitate["household_group"] = (
        cohabitate["household_group"] + married["household_group"].max() + 1
    )

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)

    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])]

    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    # married = final[final["new_mar"]==1].copy()
    # cohabitation = final[final["new_mar"]==2].copy()
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[
        final[CONDITION_4]["partner"].values, "new_household_id"
    ] = new_household_ids

    # print("Finished Pairing")
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())

    final["hh_new_id"] = np.where(
        final["stay"].isin([1]),
        final["household_id"],
        np.where(
            final["stay"].isin([0]),
            final["partner_house"],
            final["new_household_id"] + current_max_id + 1,
        ),
    )

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)][
        "household_id"
    ].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]

    households_restructuring = p_df.loc[
        p_df["household_id"].isin(household_ids_reorganized)
    ]

    households_restructuring = households_restructuring.sort_values(
        by=["household_id", "earning"], ascending=False
    )
    households_restructuring.loc[
        households_restructuring.groupby(["household_id"]).head(1).index, "relate"
    ] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        # lcm_county_id=("lcm_county_id", "first"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)

    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    p_df.update(final["MAR"])

    # print("HH SHAPE 2:", p_df["household_id"].unique().shape[0])

    new_hh = household_agg.loc[
        ~household_agg.index.isin(household_df.index.unique())
    ].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = "-1"
    household_df = pd.concat([household_df, new_hh])

    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)

    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        married_table = married_table.append(
            {
                "married": (marriage_list == 1).sum(),
                "cohabitated": (marriage_list == 2).sum(),
            },
            ignore_index=True,
        )
    orca.add_table("marriage_table", married_table)

def update_cohabitating_households(persons_df, households_df, cohabitate_list, metadata):
    """
    Updates the persons and households dataframes to reflect changes due to cohabitation events.

    This function processes a list of households undergoing cohabitation changes, updating the 
    `persons_df` and `households_df` to reflect changes in household composition and individual 
    statuses. It assigns new household IDs to individuals leaving their current households and 
    updates relationship statuses and other demographic attributes.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including household IDs 
                                   and relationship status.
        households_df (pd.DataFrame): DataFrame containing household-level data, indexed by household IDs.
        cohabitate_list (pd.Series): Series indicating households undergoing cohabitation changes.
        metadata (pd.DataFrame): DataFrame containing metadata, including the maximum household ID.

    Returns:
        tuple: Updated `persons_df` and `households_df` reflecting the cohabitation events.
    """
    # persons_df = orca.get_table("persons").local
    # persons_local_cols = persons_df.columns
    # households_df = orca.get_table("households").local
    # hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df = households_df[["lcm_county_id"]].copy()
    hh_df = hh_df.reset_index()
    households_local_cols = households_df.columns
    married_hh = cohabitate_list.index[cohabitate_list == 2].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].to_list()

    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"] == 13),"relate",] = 1
    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"].isin([1, 0])),"MAR"] = 1

    persons_df = (persons_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id"))

    leaving_person_index = persons_df.index[(persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)]

    leaving_house = persons_df.loc[leaving_person_index].copy()

    leaving_house["relate"] = 0

    persons_df = persons_df.drop(leaving_person_index)

    # Update characteristics for households staying
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    households_new = persons_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_head", "sum"),age_of_head=("age_head", "sum"), workers=("worker", "sum"),hispanic_status_of_head=("hispanic", "sum"),persons=("person", "sum"),children=("child", "sum"),seniors=("senior", "sum"),gt55=("age_gt55", "sum"),
    )

    households_new["hh_age_of_head"] = np.where(households_new["age_of_head"] < 35,"lt35",np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    households_new["hispanic_head"] = np.where( households_new["hispanic_status_of_head"] == 1, "yes", "no")
    households_new["hh_children"] = np.where( households_new["children"] >= 1, "yes", "no")
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(households_new["income"] < 30000,"lt30",np.where(households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )
    households_df.update(households_new)

    max_hh_id = metadata.loc["max_hh_id", "value"]
    # Create household characteristics for new households formed
    leaving_house["household_id"] = (
        np.arange(len(breakup_hh))
        + max(max_hh_id, households_df.index.max())
        + 1
    )
    leaving_house["person"] = 1
    leaving_house["is_head"] = np.where(leaving_house["relate"] == 0, 1, 0)
    leaving_house["race_head"] = leaving_house["is_head"] * leaving_house["race_id"]
    leaving_house["age_head"] = leaving_house["is_head"] * leaving_house["age"]
    leaving_house["hispanic_head"] = (
        leaving_house["is_head"] * leaving_house["hispanic"]
    )
    leaving_house["child"] = np.where(leaving_house["relate"].isin([2, 3, 4, 14]), 1, 0)
    leaving_house["senior"] = np.where(leaving_house["age"] >= 65, 1, 0)
    leaving_house["age_gt55"] = np.where(leaving_house["age"] >= 55, 1, 0)

    households_new = leaving_house.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )
    households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    households_new["hh_cars"] = np.where(
        households_new["cars"] == 0,
        "none",
        np.where(households_new["cars"] == 1, "one", "two or more"),
    )
    households_new["tenure"] = "unknown"
    households_new["recent_mover"] = "unknown"
    households_new["sf_detached"] = "unknown"
    households_new["tenure_mover"] = "unknown"
    households_new["block_id"] = "-1"
    households_new["hh_type"] = "-1"
    households_df = pd.concat([households_df, households_new])

    persons_df = pd.concat([persons_df, leaving_house])
    
    return persons_df, households_df

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

    divorce_households = households_df[households_df["divorced"] == 1].copy()
    DIVORCED_HOUSEHOLDS_ID = divorce_households.index.to_list()

    sizes = persons_df[persons_df["household_id"].isin(divorce_list.index) & (persons_df["relate"].isin([0, 1]))].groupby("household_id").size()


    persons_divorce = persons_df[
        persons_df["household_id"].isin(divorce_households.index)
    ].copy()


    divorced_parents = persons_divorce[
        (persons_divorce["relate"].isin([0, 1])) & (persons_divorce["MAR"] == 1)
    ].copy()

    leaving_house = divorced_parents.groupby("household_id").sample(n=1)

    staying_house = persons_divorce[~(persons_divorce.index.isin(leaving_house.index))].copy()

    # metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # give the people leaving a new household id, update their marriage status, and other variables
    leaving_house["relate"] = 0
    leaving_house["MAR"] = 3
    leaving_house["member_id"] = 1
    leaving_house["household_id"] = (
        np.arange(leaving_house.shape[0]) + max_hh_id + 1
    )

    # modify necessary variables for members staying in household
    staying_house["relate"] = np.where(
        staying_house["relate"].isin([1, 0]), 0, staying_house["relate"]
    )
    staying_house["member_id"] = np.where(
        staying_house["member_id"] != 1,
        staying_house["member_id"] - 1,
        staying_house["relate"],
    )
    staying_house["MAR"] = np.where(
        staying_house["MAR"] == 1, 3, staying_house["MAR"]
    )

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    staying_households = staying_house.copy()
    staying_households["person"] = 1
    staying_households["is_head"] = np.where(staying_households["relate"] == 0, 1, 0)
    staying_households["race_head"] = (
        staying_households["is_head"] * staying_households["race_id"]
    )
    staying_households["age_head"] = (
        staying_households["is_head"] * staying_households["age"]
    )
    staying_households["hispanic_head"] = (
        staying_households["is_head"] * staying_households["hispanic"]
    )
    staying_households["child"] = np.where(
        staying_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    staying_households["senior"] = np.where(staying_households["age"] >= 65, 1, 0)
    staying_households["age_gt55"] = np.where(staying_households["age"] >= 55, 1, 0)

    staying_households = staying_households.sort_values(by=["household_id", "relate"])
    staying_household_agg = staying_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    staying_household_agg["gt55"] = np.where(
        staying_household_agg["persons_age_gt55"] > 0, 1, 0
    )
    staying_household_agg["gt2"] = np.where(staying_household_agg["persons"] > 2, 1, 0)

    staying_household_agg["hh_workers"] = np.where(
        staying_household_agg["workers"] == 0,
        "none",
        np.where(staying_household_agg["workers"] == 1, "one", "two or more"),
    )
    staying_household_agg["hh_age_of_head"] = np.where(
        staying_household_agg["age_of_head"] < 35,
        "lt35",
        np.where(staying_household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    staying_household_agg["hh_race_of_head"] = np.where(
        staying_household_agg["race_of_head"] == 1,
        "white",
        np.where(
            staying_household_agg["race_of_head"] == 2,
            "black",
            np.where(
                staying_household_agg["race_of_head"].isin([6, 7]), "asian", "other"
            ),
        ),
    )
    staying_household_agg["hispanic_head"] = np.where(
        staying_household_agg["hispanic_head"] == 1, "yes", "no"
    )
    staying_household_agg["hh_size"] = np.where(
        staying_household_agg["size"] == 1,
        "one",
        np.where(
            staying_household_agg["size"] == 2,
            "two",
            np.where(staying_household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    staying_household_agg["hh_children"] = np.where(
        staying_household_agg["children"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_seniors"] = np.where(
        staying_household_agg["seniors"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_income"] = np.where(
        staying_household_agg["income"] < 30000,
        "lt30",
        np.where(
            staying_household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                staying_household_agg["income"] < 100,
                "gt60-lt100",
                np.where(staying_household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    staying_household_agg.index.name = "household_id"

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    new_households = leaving_house.copy()
    new_households["person"] = 1
    new_households["is_head"] = np.where(new_households["relate"] == 0, 1, 0)
    new_households["race_head"] = new_households["is_head"] * new_households["race_id"]
    new_households["age_head"] = new_households["is_head"] * new_households["age"]
    new_households["hispanic_head"] = (
        new_households["is_head"] * new_households["hispanic"]
    )
    new_households["child"] = np.where(
        new_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    new_households["senior"] = np.where(new_households["age"] >= 65, 1, 0)
    new_households["age_gt55"] = np.where(new_households["age"] >= 55, 1, 0)

    new_households = new_households.sort_values(by=["household_id", "relate"])
    household_agg = new_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["sf_detached"] = "unknown"
    household_agg["serialno"] = "unknown"
    household_agg["tenure"] = "unknown"
    household_agg["tenure_mover"] = "unknown"
    household_agg["recent_mover"] = "unknown"
    household_agg["cars"] = np.random.choice([0, 1], size=household_agg.shape[0])

    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hispanic_status_of_head"] = np.where(
        household_agg["hispanic_head"] == "yes", 1, 0
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_agg["hh_cars"] = np.where(
        household_agg["cars"] == 0,
        "none",
        np.where(household_agg["cars"] == 1, "one", "two or more"),
    )
    household_agg["block_id"] = "-1"
    household_agg["lcm_county_id"] = "-1"
    household_agg["hh_type"] = 1
    household_agg["household_type"] = 1
    household_agg["serialno"] = "-1"
    household_agg["birth"] = -99
    household_agg["divorced"] = -99

    households_df.update(staying_household_agg)

    hh_ids_p_table = np.hstack((staying_house["household_id"].unique(), leaving_house["household_id"].unique()))
    df_p = persons_df.combine_first(staying_house)
    df_p = df_p.combine_first(leaving_house)
    hh_ids_hh_table = np.hstack((households_df.index, household_agg.index))

    # merge all in one persons and households table
    new_households = pd.concat([households_df, household_agg])
    persons_df.update(staying_house)
    persons_df.update(leaving_house)

    return persons_df, new_households


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

def fetch_marriage_eligible_persons(persons_df):
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

def get_cohabitation_to_x_eligible_households(persons_df):
    """
    Identifies households eligible for the cohabitation_to_x model.

    This function filters the `persons_df` DataFrame to find individuals who are 
    currently in a cohabitation relationship (indicated by `relate` value of 13), 
    are not married (`MAR` not equal to 1), and are at least 15 years old. It 
    returns a list of unique household IDs that meet these criteria, making them 
    eligible for the cohabitation_to_x model.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, 
                                   including relationship status and age.

    Returns:
        np.ndarray: An array of unique household IDs eligible for the cohabitation_to_x model.
    """
    eligible_households = (
        persons_df[(persons_df["relate"] == 13) & \
                (persons_df["MAR"]!=1) & \
                ((persons_df["age"]>=15))]["household_id"].unique().astype(int)
    )
    return eligible_households