import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
import openmatrix as omx
import orca
import pandas as pd
import yaml


def update_education_status(persons, student_list, year):
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
    persons_df = persons.to_frame(columns=["age", "household_id", "edu", "student", "stop"])
    persons_df["stop"] = student_list
    persons_df["stop"].fillna(2, inplace=True)

    # Update education level for individuals staying in school
    weights = persons_df["edu"].value_counts(normalize=True)

    persons_df.loc[persons_df["age"] == 3, "edu"] = 2
    persons_df.loc[persons_df["age"].isin([4, 5]), "edu"] = 4

    dropping_out = persons_df.loc[persons_df["stop"] == 1].copy()
    staying_school = persons_df.loc[persons_df["stop"] == 0].copy()

    dropping_out.loc[:, "student"] = 0
    staying_school.loc[:, "student"] = 1

    # high school and high school graduates proportions
    hs_p = persons_df[persons_df["edu"].isin([15, 16])]["edu"].value_counts(
        normalize=True
    )
    hs_grad_p = persons_df[persons_df["edu"].isin([16, 17])]["edu"].value_counts(
        normalize=True
    )
    # Students all the way to grade 10
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].between(4, 13, inclusive="both"),
        staying_school["edu"] + 1,
        staying_school["edu"],
    )
    # Students in grade 11 move to either 15 or 16 based on weights
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 14,
        np.random.choice([15, 16], p=[hs_p[15], hs_p[16]]),
        staying_school["edu"],
    )
    # Students in grade 12 either get hs degree or GED
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 15,
        np.random.choice([16, 17], p=[hs_grad_p[16], hs_grad_p[17]]),
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
    # Others to be added here.

    persons_df.update(staying_school)
    persons_df.update(dropping_out)

    return persons_df


def calibrate_model(model, target_count, threshold=0.05):
    """Function to calibrate a model by adjusting the
    alternative specific constant

    Args:
        model (Urbansim Templates object): Urbansim templates object of model
        target_count (float): Target to calibrate model coefficient to
        threshold (float, optional): Error threshold for calibration. Defaults to 0.05.

    Returns:
        pd.Series: predicted outcomes from calibrated model
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


def sample_income(mean, std):
    return np.random.lognormal(mean, std)

def update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year):
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
    # Pull Data
    persons_df = orca.get_table("persons").local
    persons_cols = orca.get_injectable("persons_local_cols")
    households_df = orca.get_table("households").local
    households_cols = orca.get_injectable("households_local_cols")
    income_summary = orca.get_table("income_dist").local

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

    # Function to sample income from a normal distribution
    # Sample income for each individual based 
    # on their age and education group
    persons_df = persons_df.reset_index().merge(income_summary,
                                                on=['age_group', 'education_group'],
                                                how='left').set_index("person_id")
    persons_df['new_earning'] = persons_df.apply(lambda row: sample_income(row['mu'], row['sigma']), axis=1)

    persons_df["exit_workforce"] = exit_workforce_list
    persons_df["exit_workforce"].fillna(2, inplace=True)

    persons_df["remain_unemployed"] = stay_unemployed_list
    persons_df["remain_unemployed"].fillna(2, inplace=True)

    # Update education levels
    persons_df["worker"] = np.where(persons_df["exit_workforce"]==1, 0, persons_df["worker"])
    persons_df["worker"] = np.where(persons_df["remain_unemployed"]==0, 1, persons_df["worker"])

    persons_df["work_at_home"] = persons_df["work_at_home"].fillna(0)

    persons_df.loc[persons_df["exit_workforce"]==1, "earning"] = 0
    persons_df["earning"] = np.where(persons_df["remain_unemployed"]==0,
                                     persons_df["new_earning"], persons_df["earning"])

    # TODO: Similarly, do something for work from home
    agg_households = persons_df.groupby("household_id").agg(
        sum_workers = ("worker", "sum"),
        income = ("earning", "sum")
    )
    
    agg_households["hh_workers"] = np.where(
        agg_households["sum_workers"] == 0,
        "none",
        np.where(agg_households["sum_workers"] == 1, "one", "two or more"))
          
    # TODO: Make sure that the actual workers don't get restorted due to difference in indexing
    # TODO: Make sure there is a better way to do this
    households_df.update(agg_households)

    workers = persons_df[persons_df["worker"] == 1]
    exiting_workforce_df = orca.get_table("exiting_workforce").to_frame()
    entering_workforce_df = orca.get_table("entering_workforce").to_frame()
    if entering_workforce_df.empty:
        entering_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
    else:
        entering_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
        entering_workforce_df = pd.concat([entering_workforce_df, entering_workforce_df_new])

    if exiting_workforce_df.empty:
        exiting_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
    else:
        exiting_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
        exiting_workforce_df = pd.concat([exiting_workforce_df, exiting_workforce_df_new])
        
    orca.add_table("entering_workforce", entering_workforce_df)
    orca.add_table("exiting_workforce", exiting_workforce_df)
    orca.add_table("persons", persons_df[persons_cols])
    orca.add_table("households", households_df[households_cols])


def update_households_after_kids(persons, households, kids_moving):
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
    persons_df = orca.get_table("persons").local

    persons_local_cols = persons_df.columns

    households_df = orca.get_table("households").local
    households_local_cols = households_df.columns
    hh_id = orca.get_table("households").to_frame(columns=["lcm_county_id"]).reset_index()

    persons_df = (persons_df.reset_index().merge(hh_id, on=["household_id"]).set_index("person_id"))

    persons_df["moveoutkid"] = kids_moving

    highest_index = households_df.index.max()
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]    
    current_max_household_id = max(max_hh_id, highest_index)

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    single_per_household = (
        persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
        == 1
    )
    single_per_nonmoving = single_per_household[single_per_household == True].index.unique()
    persons_df["moveoutkid"] = np.where(persons_df["household_id"].isin(single_per_nonmoving), 0, persons_df["moveoutkid"],)

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    entire_household_moving = (
        persons_df[persons_df.index.isin(persons_df[persons_df["moveoutkid"] == 1].index.unique())]
        .groupby("household_id")
        .size()
        == persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
    )

    hh_nonmoving = entire_household_moving[entire_household_moving == True].index.unique()
    persons_df["moveoutkid"] = np.where(persons_df["household_id"].isin(hh_nonmoving), 0, persons_df["moveoutkid"])

    persons_df.loc[persons_df["moveoutkid"] == 1, "household_id"] = (np.arange(persons_df["moveoutkid"].sum()) + current_max_household_id + 1)

    new_hh = persons_df.loc[persons_df["moveoutkid"] == 1].copy()

    persons_df = persons_df.drop(persons_df[persons_df["moveoutkid"] == 1].index)
    
    old_agg_households = aggregate_household_characteristics(persons_df)

    agg_households["hh_type"] = 0  # CHANGE THIS

    households_df.update(old_agg_households)

    agg_households = aggregate_household_characteristics(new_hh)
    
    agg_households["hh_type"] = 0  # CHANGE THIS

    agg_households["tenure"] = np.random.choice(households_df["tenure"].unique(), size=agg_households.shape[0])  # Needs changed
    agg_households["recent_mover"] = np.random.choice(households_df["recent_mover"].unique(), size=agg_households.shape[0])
    agg_households["sf_detached"] = np.random.choice(households_df["sf_detached"].unique(), size=agg_households.shape[0])
    agg_households["serialno"] = "-1"
    agg_households["tenure_mover"] = "-1"
    agg_households["block_id"] = "-1"

    agg_households["cars"] = np.random.choice([0, 1, 2], size=agg_households.shape[0])

    agg_households["hh_cars"] = np.where(agg_households["cars"] == 0,"none",
                                np.where(agg_households["cars"] == 1, "one", "two or more"),)
    
    households_df["birth"] = -99
    households_df["divorced"] = -99

    households_df = pd.concat(
        [households_df[households_local_cols], agg_households[households_local_cols]]
    )

    persons_df = pd.concat([persons_df[persons_local_cols], new_hh[persons_local_cols]])
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])

    metadata = orca.get_table("metadata").to_frame()
    metadata = update_metadata(metadata, persons_df, households_df)
    orca.add_table("metadata", metadata)

    # print("Updating kids moving metrics...")
    kids_moving_table = orca.get_table("kids_move_table").to_frame()
    if kids_moving_table.empty:
        kids_moving_table = pd.DataFrame(
            [kids_moving_table.sum()], columns=["kids_moving_out"]
        )
    else:
        new_kids_moving_table = pd.DataFrame(
            {"kids_moving_out": kids_moving_table.sum()}
        )
        kids_moving_table = pd.concat([kids_moving_table, 
                                       new_kids_moving_table],
                                      ignore_index=True)
    orca.add_table("kids_move_table", kids_moving_table)

def aggregate_household_characteristics(persons_df):
    # add to orca
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 7, 9, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    persons_df = persons_df.sort_values("relate")

    agg_households = persons_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic_head", "sum"),
        seniors=("senior", "sum"),
        persons=("person", "sum"),
        gt55=("age_gt55", "sum"),
        children=("child", "sum"),
    )

    agg_households["hh_age_of_head"] = np.where(agg_households["age_of_head"] < 35, "lt35",
                                          np.where(agg_households["age_of_head"] < 65, "gt35-lt65", "gt65"))
    agg_households["hh_race_of_head"] = np.where(agg_households["race_of_head"] == 1, "white",
                                           np.where(agg_households["race_of_head"] == 2, "black",
                                           np.where(agg_households["race_of_head"].isin([6, 7]), "asian", "other")))
    
    agg_households["hispanic_head"] = np.where(agg_households["hispanic_status_of_head"] == 1, "yes", "no")
    agg_households["hh_size"] = np.where(agg_households["persons"] == 1, "one",
                                   np.where(agg_households["persons"] == 2, "two",
                                   np.where(agg_households["persons"] == 3, "three", "four or more")))
    agg_households["hh_children"] = np.where(agg_households["children"] >= 1, "yes", "no")
    agg_households["hh_income"] = np.where(agg_households["income"] < 30000, "lt30",
                                     np.where(agg_households["income"] < 60, "gt30-lt60",
                                     np.where(agg_households["income"] < 100, "gt60-lt100",
                                     np.where(agg_households["income"] < 150, "gt100-lt150", "gt150"))))
    agg_households["hh_workers"] = np.where(agg_households["workers"] == 0, "none",
                                      np.where(agg_households["workers"] == 1, "one", "two or more"))
    agg_households["hh_seniors"] = np.where(agg_households["seniors"] >= 1, "yes", "no")
    agg_households["gt55"] = np.where(agg_households["age_gt55"] > 0, 1, 0)
    agg_households["gt2"] = np.where(agg_households["persons"] > 2, 1, 0)

    return agg_households

def fix_erroneous_households(persons, households):
    print("Fixing erroneous households")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns

    households_to_drop = p_df[p_df['relate'].isin([1, 13])].groupby('household_id')['relate'].nunique().reset_index()
    households_to_drop = households_to_drop[households_to_drop["relate"]==2]["household_id"].to_list()

    household_df = household_df.drop(households_to_drop)
    p_df = p_df[~p_df["household_id"].isin(households_to_drop)]

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



def update_married_households_random(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households
    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals
    Returns:
        None
    """
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

    if ((relevant["new_mar"] ==1).sum() <= 10) or ((relevant["new_mar"] ==2).sum() <= 10):
        return None

    relevant.sort_values("new_mar", inplace=True)

    relevant = relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")


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
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())
    final["hh_new_id"] = np.where(final["stay"].isin([1]), final["household_id"], np.where(final["stay"].isin([0]),final["partner_house"],final["new_household_id"] + current_max_id + 1))

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)]["household_id"].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]

    households_restructuring = p_df.loc[p_df["household_id"].isin(household_ids_reorganized)]

    households_restructuring = households_restructuring.sort_values(by=["household_id", "earning"], ascending=False)
    households_restructuring.loc[households_restructuring.groupby(["household_id"]).head(1).index, "relate"] = 0

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
    household_agg = p_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_id", "first"),age_of_head=("age", "first"),size=("person", "sum"),workers=("worker", "sum"),hispanic_head=("hispanic_head", "sum"),persons_age_gt55=("age_gt55", "sum"),seniors=("senior", "sum"),children=("child", "sum"),persons=("person", "sum"),)

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

    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    p_df["NEW_MAR"] = final["MAR"]
    p_df["MAR"] = np.where(p_df["NEW_MAR"].isna(),p_df["MAR"], p_df["NEW_MAR"])

    new_hh = household_agg.loc[~household_agg.index.isin(household_df.index.unique())].copy()
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

    metadata = orca.get_table("metadata").to_frame()
    metadata = update_metadata(metadata, p_df, household_df)

    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])
    orca.add_table("metadata", metadata)
    
    married_table = orca.get_table("marriage_table").to_frame()
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

    orca.add_table("marriage_table", married_table)

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



def update_cohabitating_households(persons, households, cohabitate_list):
    """
    Updating households and persons after cohabitation model.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_local_cols = persons_df.columns
    households_df = orca.get_table("households").local
    hh_df = households.to_frame(columns=["lcm_county_id"])
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

    households_new = aggregate_household_characteristics(persons_df)
    households_df.update(households_new)

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # Create household characteristics for new households formed
    leaving_house["household_id"] = (np.arange(len(breakup_hh)) + max(max_hh_id, households_df.index.max()) + 1)
    households_new = aggregate_household_characteristics(leaving_house)

    households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    households_new["hh_cars"] = np.where(households_new["cars"] == 0, "none", 
                                np.where(households_new["cars"] == 1, "one", "two or more"))
    households_new["tenure"] = "unknown"
    households_new["recent_mover"] = "unknown"
    households_new["sf_detached"] = "unknown"
    households_new["tenure_mover"] = "unknown"
    households_new["block_id"] = "-1"
    households_new["lcm_county_id"] = "-1"
    households_new["hh_type"] = "-1"

    households_df = pd.concat([households_df, households_new])

    persons_df = pd.concat([persons_df, leaving_house])

    metadata = update_metadata(metadata, persons_df, households_df)

    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_table("metadata", metadata)

def update_divorce(divorce_list):
    """
    Updating stats for divorced households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        divorce_list (pd.Series): pandas Series of the divorced households

    Returns:
        None
    """
    # print("Updating household stats...")
    households_local_cols = orca.get_table("households").local.columns

    persons_local_cols = orca.get_table("persons").local.columns

    households_df = orca.get_table("households").local

    persons_df = orca.get_table("persons").local

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

    metadata = orca.get_table("metadata").to_frame()
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
    staying_household_agg = aggregate_household_characteristics(staying_households)

    staying_household_agg.index.name = "household_id"
    households_df.update(staying_household_agg)

    # give the people leaving a new household id, update their marriage status, and other variables
    leaving_house["relate"] = 0
    leaving_house["MAR"] = 3
    leaving_house["member_id"] = 1
    leaving_house["household_id"] = np.arange(leaving_house.shape[0]) + max_hh_id + 1

    # initiate new households with individuals leaving house
    # new_households = leaving_house.copy()
    leaving_household_agg = aggregate_household_characteristics(leaving_house)
    
    leaving_household_agg["block_id"] = "-1"
    leaving_household_agg["lcm_county_id"] = "-1"
    leaving_household_agg["hh_type"] = 1
    leaving_household_agg["household_type"] = 1
    leaving_household_agg["serialno"] = "-1"
    leaving_household_agg["birth"] = -99
    leaving_household_agg["divorced"] = -99
    leaving_household_agg["sf_detached"] = "unknown"
    leaving_household_agg["serialno"] = "unknown"
    leaving_household_agg["tenure"] = "unknown"
    leaving_household_agg["tenure_mover"] = "unknown"
    leaving_household_agg["recent_mover"] = "unknown"
    leaving_household_agg["cars"] = np.random.choice([0, 1], size=leaving_household_agg.shape[0])
    leaving_household_agg["hh_cars"] = np.where(leaving_household_agg["cars"] == 0, "none",
                                       np.where(leaving_household_agg["cars"] == 1, "one", "two or more"))
    # hh_ids_p_table = np.hstack((staying_house["household_id"].unique(), leaving_house["household_id"].unique()))

    df_p = persons_df.combine_first(staying_house[persons_local_cols])
    df_p = df_p.combine_first(leaving_house[persons_local_cols])

    # hh_ids_hh_table = np.hstack((households_df.index, household_agg.index))

    # merge all in one persons and households table
    households_df = pd.concat([households_df[households_local_cols], leaving_household_agg[households_local_cols]])

    # persons_df.update(staying_house[persons_local_cols])
    # persons_df.update(leaving_house[persons_local_cols])

    metadata = orca.get_table("metadata").to_frame()
    metadata = update_metadata(metadata, persons_df, households_df)

    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_table("metadata", metadata)

    # print("Updating divorce metrics...")
    divorce_table = orca.get_table("divorce_table").to_frame()
    if divorce_table.empty:
        divorce_table = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
    else:
        new_divorce = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
        divorce_table = pd.concat([divorce_table, new_divorce], ignore_index=True)
    
    orca.add_table("divorce_table", divorce_table) 

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

def identify_dead_households(persons_df):
    """Function to identify households with everyone deceased

    Args:
        persons_df (pd.DataFrame): Pandas DataFrame of persons data

    Returns:
        list: List of household ids where everyone has passed
    """
    persons_df["member"] = 1
    dead_fraction = persons_df.groupby("household_id").agg(
        num_dead=("dead", "sum"), size=("member", "sum")
    )
    return dead_fraction[dead_fraction["num_dead"] == dead_fraction["size"]].index.to_list()

# Mortality model returns a list of 0s representing alive and 1 representing dead
# Then adds that list to the persons table and updates persons and households tables accordingly
def remove_dead_persons(persons, households, fatality_list, year):
    """
    This function updates the persons table from the output of the fatality model.
    Takes in the persons and households orca tables.

    Args:
        persons (DataFramWrapper): DataFramWrapper of persons table
        households (DataFramWrapper): DataFramWrapper of households table
        fatality_list (pd.Series): Pandas Series of fatality list
    """

    houses = households.local
    households_columns = orca.get_injectable("households_local_cols")

    # Pulling the persons data
    persons_df = persons.local
    persons_columns = orca.get_injectable("persons_local_cols")

    persons_df["dead"] = -99
    persons_df["dead"] = fatality_list
    graveyard = persons_df[persons_df["dead"] == 1].copy()

    # HOUSEHOLD WHERE EVERYONE DIES
    dead_households = identify_dead_households(persons_df)
    grave_persons = persons_df[persons_df["household_id"].isin(dead_households)].copy()
    # Drop out of the persons table
    persons_df = persons_df.loc[~persons_df["household_id"].isin(dead_households)]
    # Drop out of the households table
    houses = houses.drop(dead_households)

    ##################################################
    ##### HOUSEHOLDS WHERE PART OF HOUSEHOLD DIES ####
    ##################################################
    dead, alive = persons_df[persons_df["dead"] == 1].copy(), persons_df[persons_df["dead"] == 0].copy()

    #################################
    # Alive heads, Dead partners
    #################################
    # Dead partners, either married or cohabitating
    dead_partners = dead[dead["relate"].isin([1, 13])]  # This will need changed
    # Alive heads
    alive_heads = alive[alive["relate"] == 0]
    alive_heads = alive_heads[["household_id", "MAR"]]
    widow_heads = alive[(alive["household_id"].isin(dead_partners["household_id"])) & (alive["relate"]==0)].copy()
    widow_heads["MAR"].values[:] = 3

    # Dead heads, alive partners
    dead_heads = dead[dead["relate"] == 0]
    alive_partner = alive[alive["relate"].isin([1, 13])].copy()
    alive_partner = alive_partner[
        ["household_id", "MAR"]
    ]  ## Pull the relate status and update it
    widow_partners = alive[(alive["household_id"].isin(dead_heads["household_id"])) & (alive["relate"].isin([1, 13]))].copy()

    widow_partners["MAR"].values[:] = 3  # THIS MIGHT NEED VERIFICATION, WHAT DOES MAR MEAN?

    # Merge the two groups of widows
    widows = pd.concat([widow_heads, widow_partners])[["MAR"]]
    # Update the alive database's MAR values using the widows table
    alive_copy = alive.copy()
    alive.loc[widows.index, "MAR"] = 3
    alive["MAR"] = alive["MAR"].astype(int)


    # Select the households in alive where the heads died
    alive_sort = alive[alive["household_id"].isin(dead_heads["household_id"])].copy()
    alive_sort["relate"] = alive_sort["relate"].astype(int)

    if len(alive_sort.index) > 0:
        alive_sort.sort_values("relate", inplace=True)
        # Restructure all the households where the head died
        alive_sort = alive_sort[["household_id", "relate", "age"]]
        # print("Starting to restructure household")
        # Apply the rez function
        alive_sort = alive_sort.groupby("household_id").apply(rez)

        # Update relationship values and make sure correct datatype is used
        alive.loc[alive_sort.index, "relate"] = alive_sort["relate"]
        alive["relate"] = alive["relate"].astype(int)

    alive["is_relate_0"] = (alive["relate"]==0).astype(int)
    alive["is_relate_1"] = (alive["relate"]==1).astype(int)

    alive_agg = alive.groupby("household_id").agg(sum_relate_0 = ("is_relate_0", "sum"), sum_relate_1 = ("is_relate_1", "sum"))
    
    # Dropping households with more than one head or more than one partner
    alive_agg = alive_agg[(alive_agg["sum_relate_1"]<=1) & (alive_agg["sum_relate_0"]<=1)]
    alive_hh = alive_agg.index.tolist()
    alive = alive[alive["household_id"].isin(alive_hh)]

    households_new = aggregate_household_characteristics(alive)

    houses.update(households_new)
    houses = houses.loc[alive_hh]

    graveyard_table = orca.get_table("pop_over_time").to_frame()
    if graveyard_table.empty:
        dead_people = grave_persons.copy()
    else:
        dead_people = pd.concat([graveyard_table, grave_persons])

    orca.add_table("persons", alive[persons_columns])
    orca.add_table("households", houses[households_columns])
    orca.add_table("graveyard", dead_people[persons_columns])

    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local

    metadata = orca.get_table("metadata").to_frame()
    metata = update_metadata(metadata, persons_df, households_df)
    orca.add_table("metadata", metadata)

def rez(group):
    """
    Function to change the household head role
    TODO: This needs to become vectorized to make it faster.
    """
    # Update the relate variable for the group
    if group["relate"].iloc[0] == 1:
        group["relate"].iloc[0] = 0
        return group
    if 13 in group["relate"].values:
        group["relate"].replace(13, 0, inplace=True)
        return group

    # Get the maximum age of the household, oldest person becomes head of household
    # Verify this with Juan.
    new_head_idx = group["age"].idxmax()
    # Function to map the relation of new head
    map_func = create_household_role_mapping(group.loc[new_head_idx, "relate"])
    group.loc[new_head_idx, "relate"] = 0
    # breakpoint()
    group.relate = group.relate.map(map_func)
    return group

def create_household_role_mapping(old_role):
    """
    Function that uses the relationship mapping in the
    provided table and returns a function that maps
    new household roles.

    Returns:
    Function that takes a persons old role
    # and gives them a new one based on how the 
    # household is restructured
    """

    string_old_role = str(old_role)
 
    def inner(role):
        rel_map = orca.get_table("rel_map").to_frame()
        if role == 0:
            new_role = 0
        else:
            new_role = rel_map.loc[role, string_old_role]
        return new_role

    return inner

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
    
    metadata = update_metadata(metadata, persons_df, households_df)

    orca.add_table("persons", persons_df.loc[:, persons_local_cols])
    orca.add_table("households", households_df.loc[:, households_local_cols])
    orca.add_table("metadata", metadata)

def update_metadata(metadata, persons_df, households_df):
    """Function to update the metadata
    for persons and households indices.

    Args:
        metadata (pd.DataFrame): Dataframe of the indices metadata
        persons_df (pd.DataFrame): Dataframe of the persons table
        households_df (pd.DataFrame): Dataframe of the households table

    Returns:
        pd.DataFrame: Updated metadata dataframe
    """
    metadata_max_households_id = metadata.loc["max_hh_id", "value"]
    metadata_max_persons_id = metadata.loc["max_p_id", "value"]
    
    max_households_id = households_df.index.max()
    max_person_id = persons_df.index.max()


    if max_households_id > metadata_max_households_id:
        metadata.loc["max_hh_id", "value"] = max_households_id
    
    if max_person_id > metadata_max_persons_id:
        metadata.loc["max_p_id", "value"] = max_person_id
    
    return metadata

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
