import pandas as pd
import numpy as np

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