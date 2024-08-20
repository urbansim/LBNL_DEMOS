import numpy as np
import pandas as pd
import orca
from scipy.special import softmax

def simulation_mnl(data, coeffs):
    """Function to run simulation of the MNL model

    Args:
        data (_type_): _description_
        coeffs (_type_): _description_

    Returns:
        Pandas Series: Pandas Series of the outcomes of the simulated model
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


# -----------------------------------------------------------------------------------------
# BIRTH TABLE
# -----------------------------------------------------------------------------------------
def update_birth_eligibility_count_table(btable_elig_df, eligible_household_ids, year):
    """
    Function to update the birth eligibility count table
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
    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
    )

    # Subset of eligible households
    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()
    eligible_household_ids = households_df.loc[ELIGIBLE_HH].index.to_list()
    return eligible_household_ids

def update_birth(persons_df, households_df, birth_list):
    """
    Update the persons tables with newborns and household sizes

    Args:   
        persons_df (pd.DataFrame): DataFrame of the persons table
        households_df (pd.DataFrame): DataFrame of the households table
        birth_list (pd.Series): Pandas Series of the households with newborns

    Returns:
        pd.DataFrame: DataFrame of the persons table
        pd.DataFrame: DataFrame of the households table
    """

    # Pull max person index from persons table
    highest_index = persons_df.index.max()

    # Check if the pop_over_time is an empty dataframe
    grave = orca.get_table("pop_over_time").to_frame()

    # If not empty, update the highest index with max index of all people
    metadata = orca.get_table("metadata").to_frame()

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
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    return metadata

def update_income(persons_df, households_df, income_rates, year):
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
    new_workforce_df = pd.DataFrame(
            data={"year": [year], "entering_workforce": [persons_df[persons_df["remain_unemployed"]==0].shape[0]],
            "exiting_workforce": [persons_df[persons_df["exit_workforce"]==1].shape[0]]})
    if workforce_df.empty:
        workforce_df = new_workforce_df
    else:
        workforce_df = pd.concat([workforce_df, new_workforce_df])
    return workforce_df

def aggregate_household_labor_variables(persons_df, households_df):
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
    return np.random.lognormal(mean, std)

def update_labor_status(persons_df, stay_unemployed_list, exit_workforce_list, income_summary):
    """
    Function to update the worker status in persons table based
    on the labor participation model

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    #####################################################
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    education_intervals = [0, 18, 22, 200]
    # Define the labels for age and education groups
    age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
    education_labels = ['lte17', '18-21', 'gte22']
    # Create age and education groups with labels
    persons_df['age_group'] = pd.cut(persons_df['age'], bins=age_intervals, labels=age_labels, include_lowest=True)
    persons_df['education_group'] = pd.cut(persons_df['edu'], bins=education_intervals, labels=education_labels, include_lowest=True)
    #####################################################

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


def extract_students(persons):
    """Retrieve the list of grade school students


    Args:
        persons (Orca table): Persons orca table

    Returns:
        DataFrame: pandas dataframe of grade school students.
    """
    edu_levels = np.arange(3, 16).astype(float)
    STUDENTS_CONDITION = (persons["student"]==1) & (persons["edu"].isin(edu_levels))
    students_df = persons[STUDENTS_CONDITION].copy()
    students_df["STUDENT_SCHOOL_LEVEL"] = np.where(
        students_df["edu"]<=8, "ELEMENTARY",
        np.where(students_df["edu"]<=11, "MIDDLE", "HIGH"))
    students_df = students_df.reset_index()
    return students_df

def create_student_groups(students_df):
    """Generate a dataframe of students from same
    household and same grade level

    Args:
        students_df (DataFrame): grade level students DataFrame

    Returns:
        DataFrame: grouped students by household and grade level
    """
    students_df = students_df[students_df["DET_DIST_TYPE"]==students_df["STUDENT_SCHOOL_LEVEL"]].reset_index(drop=True)
    student_groups = students_df.groupby(['household_id', 'GEOID10_SD', 'STUDENT_SCHOOL_LEVEL'])['person_id'].apply(list).reset_index(name='students')
    student_groups["DISTRICT_LEVEL"] = student_groups.apply(lambda row: (row['GEOID10_SD'], row['STUDENT_SCHOOL_LEVEL']), axis=1)
    student_groups["size_student_group"] = [len(x) for x in student_groups["students"]]
    student_groups = student_groups.sort_values(by=["GEOID10_SD", "STUDENT_SCHOOL_LEVEL"]).reset_index(drop=True)
    student_groups["CUM_STUDENTS"] = student_groups.groupby(["GEOID10_SD", "STUDENT_SCHOOL_LEVEL"])["size_student_group"].cumsum()
    return student_groups

def assign_schools(student_groups, blocks_districts, schools_df):
    """Assigns students to schools from the same school district

    Args:
        student_groups (DataFrame): Dataframe of student groups
        blocks_districts (DataFrame): Dataframe of crosswalk between
        blocks and school districts
        schools_df (DataFrame): DataFrame of list of schools in region

    Returns:
        list: list of assigned students
    """
    assigned_students_list = []
    for tuple in blocks_districts["DISTRICT_LEVEL"].unique():
        SCHOOL_DISTRICT = tuple[0]
        SCHOOL_LEVEL = tuple[1]
        schools_pool = schools_df[(schools_df["SCHOOL_LEVEL"]==SCHOOL_LEVEL) &\
                                (schools_df["GEOID10_SD"]==SCHOOL_DISTRICT)].copy()
        student_pool = student_groups[(student_groups["STUDENT_SCHOOL_LEVEL"]==SCHOOL_LEVEL) &\
                        (student_groups["GEOID10_SD"]==SCHOOL_DISTRICT)].copy()
        student_pool = student_pool.sample(frac = 1)
        # Iterate over schools_df
        for idx, row in schools_pool.iterrows():
            # Get the pool of students for the district and level of the school
            SCHOOL_LEVEL = row["SCHOOL_LEVEL"]
            SCHOOL_DISTRICT = row["GEOID10_SD"]
            # Calculate the number of students to assign
            n_students = min(student_pool["size_student_group"].sum(), row['CAP_TOTAL'])
            student_pool["CUM_STUDENTS"] = student_pool["size_student_group"].cumsum()
            student_pool["ASSIGNED"] = np.where(student_pool["CUM_STUDENTS"]<=n_students, 1, 0)
            # Randomly sample students without replacement
            assigned_students = student_pool[student_pool["CUM_STUDENTS"]<=n_students].copy()
            assigned_students["SCHOOL_ID"] = row["SCHOOL_ID"]
            assigned_students_list.append(assigned_students)
            student_pool = student_pool[student_pool["ASSIGNED"]==0].copy()
    return assigned_students_list

def create_results_table(students_df, assigned_students_list, year):
    """Creates table of student assignment

    Args:
        students_df (DataFrame): students dataframe
        assigned_students_list (list): student assignment list
        year (int): year of simulation

    Returns:
        DataFrame: student assignment dataframe
    """
    assigned_students_df = pd.concat(assigned_students_list)[["students", "household_id", "GEOID10_SD", "STUDENT_SCHOOL_LEVEL", "SCHOOL_ID"]]
    assigned_students_df = assigned_students_df.explode("students").rename(columns={"students": "person_id",
                                                                                    "SCHOOL_ID": "school_id",})
    school_assignment_df = students_df[["person_id"]].merge(assigned_students_df[["person_id", "school_id", "GEOID10_SD"]], on="person_id", how='left').fillna("-1")
    school_assignment_df["year"] = year
    return school_assignment_df

def export_demo_table(table_name):
    """
    Export the tables

    Args:
        table_name (string): Name of the orca table
    """
    
    region_code = orca.get_injectable("region_code")
    output_folder = orca.get_injectable("output_folder")
    df = orca.get_table(table_name).to_frame()
    csv_name = table_name + "_" + region_code +".csv"
    df.to_csv(output_folder+csv_name, index=False)

def deduplicate_updated_households(updated, persons_df, metadata):
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

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        kids_moving (pd.Series): Pandas Series of kids moving out of household

    Returns:
        None
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