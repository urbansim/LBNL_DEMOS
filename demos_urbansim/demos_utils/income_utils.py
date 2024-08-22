import pandas as pd

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