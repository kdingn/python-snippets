import numpy as np
import pandas as pd

def commaConcatenatedStringColumn_to_elementList(df,col):
    """
    Creating list of category elements
    from comma concatenated string column in dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for applying this function.
    col : string
        Column for applying this function.

    Returns
    ----------
    list
        List of values in the column of the dataframe.
    """
    uniques = list(df[col].unique())
    uniques.remove(np.nan)
    allel = []
    for el in list(map(lambda x : x.split(", "), uniques)):
        allel.extend(el)
    return list(set(allel))


def impute_by_group(
    df,
    groupkeys,
    impute_col,
    target_col
):
    """
    Impute by most frequent values in the same groups.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for applying this function.
    groupkeys : list
        Column names of the groups.
    impute_col : string
        Column name for imputing.
    target_col : string
        Column name which does not include N/A value.
        
    Returns
    ----------
    pandas.DataFrame
        Dataframe whose impute_col is imputed.
    """
    groupkeys_icol = groupkeys.copy()
    groupkeys_icol.append(impute_col)
    groupkeys_icol_tcol = groupkeys_icol.copy()
    groupkeys_icol_tcol.append(target_col)
    groupkeys_count = groupkeys.copy()
    groupkeys_count.append("count")

    dfcount = df.groupby(
        groupkeys_icol,as_index=False
    ).count()[groupkeys_icol_tcol].rename(
        columns = {target_col:"count"}
    )
    dfcount = pd.merge(
        dfcount.groupby(
            groupkeys,as_index=False
        ).max()[groupkeys_count],
        dfcount,
        on=groupkeys_count,
        how="left"
    )
    dfcount = dfcount.drop("count",axis=1).drop_duplicates(
        groupkeys
    ).rename(columns={impute_col:f"{impute_col}_impute"})

    dfnotnull = df[~df[impute_col].isnull()].copy()

    dfnull = df[df[impute_col].isnull()].copy()
    dfnull = dfnull.merge(dfcount,on=groupkeys,how="left")
    dfnull[impute_col] = dfnull[f"{impute_col}_impute"]
    dfnull = dfnull.drop(f"{impute_col}_impute",axis=1)

    df = pd.concat([dfnotnull,dfnull]).reset_index(drop=True)
    return df