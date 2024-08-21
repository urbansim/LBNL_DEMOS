import numpy as np
import orca
import pandas as pd
import utils


def identify_dead_households(persons_df):
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

    alive["person"] = 1
    alive["is_head"] = np.where(alive["relate"] == 0, 1, 0)
    alive["race_head"] = alive["is_head"] * alive["race_id"]
    alive["age_head"] = alive["is_head"] * alive["age"]
    alive["hispanic_head"] = alive["is_head"] * alive["hispanic"]
    alive["child"] = np.where(alive["relate"].isin([2, 3, 4, 14]), 1, 0)
    alive["senior"] = np.where(alive["age"] >= 65, 1, 0)
    alive["age_gt55"] = np.where(alive["age"] >= 55, 1, 0)

    households_new = alive.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
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

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
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
    map_func = produce_map_func(group.loc[new_head_idx, "relate"])
    group.loc[new_head_idx, "relate"] = 0
    # breakpoint()
    group.relate = group.relate.map(map_func)
    return group

# TODO refactor this method so it can be cleanly used with any model
# Might need to add some stuff (update more variables) or have a flag to determine which function is calling this
def update_households(alive, dead, old_incomes, subtract=True):
    """
    Function to update the households characteristics, namely income, num of workers, race, and age of head.
    """
    # By default the dead variable represents people dying or leaving a household
    if subtract:
        alive = alive.sort_values("relate")
        dead_aggs = dead.groupby("household_id").agg({"earning": "sum"})

        aggregates = alive.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum", "race_id": "first", "age": "first"}
        )
        # print(list(orca.get_table("households").to_frame().columns))
        # TODO -- REPLACE THESE OPERATIONS BY PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", old_incomes.subtract(dead_aggs["earning"], fill_value=0)
        )
        orca.get_table("households").update_col("workers", aggregates["worker"])
        orca.get_table("households").update_col("race_of_head", aggregates["race_id"])
        orca.get_table("households").update_col("age_of_head", aggregates["age"])
    # In a certain case, the dead variable actually represents living people being added -- What is this case?
    else:
        aggregates = dead.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum"}
        )
        house_df = orca.get_table("households").to_frame(columns=["income", "workers"])
        # TODO -- REPLACE THESE OPERATIONS WITH PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", house_df["income"].add(aggregates["earning"], fill_value=0)
        )
        orca.get_table("households").update_col(
            "workers", house_df["workers"].add(aggregates["worker"], fill_value=0)
        )

    # return households

# Function that takes the head's previous role and returns a function
# that maps roles to new roles based on restructuring
def produce_map_func(old_role):
    """
    Function that uses the relationship mapping in the
    provided table and returns a function that maps
    new household roles.
    """
    # old role is the previous number of the person who has now been promoted to head of the household
    sold_role = str(old_role)
 
    def inner(role):
        rel_map = orca.get_table("rel_map").to_frame()
        if role == 0:
            new_role = 0
        else:
            new_role = rel_map.loc[role, sold_role]
        return new_role

    # Returns function that takes a persons old role and gives them a new one based on how the household is restructured
    return inner
