import numpy as np
import pandas as pd


def update_education_status(persons_df, student_list, year):
    """
    Update the education status of individuals in the persons DataFrame.

    This function updates the education level and student status of individuals 
    based on their current age and a provided list indicating whether they continue 
    their education. It differentiates between those dropping out and those staying 
    in school, and adjusts their education levels accordingly.

    Args:
        persons_df (pd.DataFrame): DataFrame containing person-level data, including 
                                   age, education, and student status.
        student_list (pd.Series): Series indicating whether individuals will continue 
                                  their education (0 for staying, 1 for dropping out).
        year (int): The current year for which the education status updates are being applied.

    Returns:
        pd.DataFrame: Updated `persons_df` with modified education levels and student status.
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

def extract_students(persons_df):
    """
    Extract students from the persons DataFrame.

    This function identifies individuals who are students based on their education level
    and student status. It creates a DataFrame of students, categorizing them into 
    elementary, middle, and high school levels.

    Args:
        persons (pd.DataFrame): DataFrame containing person-level data, including education 
                                level and student status.

    Returns:
        pd.DataFrame: A DataFrame of students with additional columns indicating their 
                      school level (elementary, middle, or high).
    """
    breakpoint()
    edu_levels = np.arange(3, 16).astype(float)
    STUDENTS_CONDITION = (persons_df["student"]==1) & (persons_df["edu"].isin(edu_levels))
    students_df = persons_df[STUDENTS_CONDITION].copy()
    students_df["student_school_level"] = np.where(students_df["edu"]<=8, "ELEMENTARY", np.where(students_df["edu"]<=11, "MIDDLE", "HIGH"))
    students_df = students_df.reset_index()
    breakpoint()
    return students_df

def create_student_groups(students_df):
    """Generate a dataframe of students from same
    household and same grade level

    Args:
        students_df (DataFrame): grade level students DataFrame

    Returns:
        DataFrame: grouped students by household and grade level
    """
    # breakpoint()
    students_df = students_df[students_df["school_level"]==students_df["student_school_level"]].reset_index(drop=True)
    # breakpoint()
    student_groups = students_df.groupby(['household_id', 'district_id', 'student_school_level'])['person_id'].apply(list).reset_index(name='students')
    # breakpoint()
    student_groups["district_level"] = student_groups.apply(lambda row: (row['district_id'], row['student_school_level']), axis=1)
    # breakpoint()
    student_groups["size_student_group"] = [len(x) for x in student_groups["students"]]
    # breakpoint()
    student_groups = student_groups.sort_values(by=["district_id", "student_school_level"]).reset_index(drop=True)
    # breakpoint()
    student_groups["cumulative_students"] = student_groups.groupby(["district_id", "student_school_level"])["size_student_group"].cumsum()
    # breakpoint()
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
    # breakpoint()
    for tuple in blocks_districts["district_level"].unique():
        # breakpoint()
        SCHOOL_DISTRICT = tuple[0]
        SCHOOL_LEVEL = tuple[1]
        # breakpoint()
        schools_pool = schools_df[(schools_df["school_level"]==SCHOOL_LEVEL) &\
                                (schools_df["distict_id"]==SCHOOL_DISTRICT)].copy()
        # breakpoint()
        student_pool = student_groups[(student_groups["student_school_level"]==SCHOOL_LEVEL) &\
                        (student_groups["distict_id"]==SCHOOL_DISTRICT)].copy()
        # breakpoint()
        student_pool = student_pool.sample(frac = 1)
        # breakpoint()
        # Iterate over schools_df
        for idx, row in schools_pool.iterrows():
            # Get the pool of students for the district and level of the school
            # breakpoint()
            SCHOOL_LEVEL = row["school_level"]
            SCHOOL_DISTRICT = row["distict_id"]
            # Calculate the number of students to assign
            # breakpoint()
            n_students = min(student_pool["size_student_group"].sum(), row['CAP_TOTAL'])
            # breakpoint()
            student_pool["cumulative_students"] = student_pool["size_student_group"].cumsum()
            # breakpoint()
            student_pool["assigned"] = np.where(student_pool["cumulative_students"]<=n_students, 1, 0)
            # breakpoint()
            # Randomly sample students without replacement
            assigned_students = student_pool[student_pool["cumulative_students"]<=n_students].copy()
            # breakpoint()
            assigned_students["school_id"] = row["school_id"]
            # breakpoint()
            assigned_students_list.append(assigned_students)
            # breakpoint()
            student_pool = student_pool[student_pool["assigned"]==0].copy()
            # breakpoint()
    return assigned_students_list

def create_results_table(students_df, assigned_students_list, year):
    """
    Create a results table for student school assignments.

    This function consolidates student assignment data into a single DataFrame,
    detailing which students have been assigned to which schools. It includes
    information about the household, school district, and school level.

    Args:
        students_df (pd.DataFrame): DataFrame of students with their IDs and other attributes.
        assigned_students_list (list): List of DataFrames, each containing student assignments 
                                       to schools, including school IDs and district information.
        year (int): The year for which the assignments are being recorded.

    Returns:
        pd.DataFrame: A DataFrame containing the results of student assignments, including 
                      student IDs, school IDs, and the year of assignment.
    """
    assigned_students_df = pd.concat(assigned_students_list)[["students", "household_id", "district_id", "student_school_level", "school_id"]]
    assigned_students_df = assigned_students_df.explode("students").rename(columns={"students": "person_id",
                                                                                    "school_id": "school_id",})
    school_assignment_df = students_df[["person_id"]].merge(assigned_students_df[["person_id", "school_id", "district_id"]], on="person_id", how='left').fillna("-1")
    school_assignment_df["year"] = year
    return school_assignment_df
