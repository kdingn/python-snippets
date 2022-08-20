import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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


# def impute_by_group(
#     df,
#     groupkeys,
#     impute_col,
#     target_col
# ):
#     """
#     Impute by most frequent values in the same groups.
#     The column value for imputation is supposed to be categorical.
#     (not numerical)
    
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Dataframe for applying this function.
#     groupkeys : list
#         Column names of the groups.
#     impute_col : string
#         Column name for imputing.
#     target_col : string
#         Column name which does not include N/A value.
        
#     Returns
#     ----------
#     pandas.DataFrame
#         Dataframe whose impute_col is imputed.
#     """
#     groupkeys_icol = groupkeys.copy()
#     groupkeys_icol.append(impute_col)
#     groupkeys_icol_tcol = groupkeys_icol.copy()
#     groupkeys_icol_tcol.append(target_col)
#     groupkeys_count = groupkeys.copy()
#     groupkeys_count.append("count")

#     dfcount = df.groupby(
#         groupkeys_icol,as_index=False
#     ).count()[groupkeys_icol_tcol].rename(
#         columns = {target_col:"count"}
#     )
#     dfcount = pd.merge(
#         dfcount.groupby(
#             groupkeys,as_index=False
#         ).max()[groupkeys_count],
#         dfcount,
#         on=groupkeys_count,
#         how="left"
#     )
#     dfcount = dfcount.drop("count",axis=1).drop_duplicates(
#         groupkeys
#     ).rename(columns={impute_col:f"{impute_col}_impute"})

#     dfnotnull = df[~df[impute_col].isnull()].copy()

#     dfnull = df[df[impute_col].isnull()].copy()
#     dfnull = dfnull.merge(dfcount,on=groupkeys,how="left")
#     dfnull[impute_col] = dfnull[f"{impute_col}_impute"]
#     dfnull = dfnull.drop(f"{impute_col}_impute",axis=1)

#     df = pd.concat([dfnotnull,dfnull]).reset_index(drop=True)
#     return df


def impute_cat_by_groupkey(
    x_train,
    impute_col,
    groupkey,
    SEED=2022,
    n_split=5
):
    """
    Impute by most frequent values in the same group.
    The column value for imputation is supposed to be categorical.
    (not numerical)
    
    Parameters
    ----------
    x_train : pandas.DataFrame
        Dataframe for applying this function.
    impute_col : string
        Column name for imputing.
    groupkey : list
        Column names of the groups.
    SEED : int, default=2022
        Seed number for KFold.
        Seed number should be the same as validation splits.
    n_split : int, default=5
        The number of splits of KFold.
        
    Returns
    ----------
    pandas.DataFrame
        Dataframe whose impute_col is imputed.
    """
    ## prepare return df
    dfreturn = pd.DataFrame()
    ## keep index
    x_train = x_train.reset_index()
    ## kfold
    kf = KFold(n_splits=n_split,shuffle=True,random_state=SEED)
    indexes = list(kf.split(x_train))
    ## keys
    columns = groupkey.copy()
    columns.append(impute_col)
    columnsall = columns.copy()
    columnsall.append("index")
    ## loop
    for train_index, valid_index in indexes:
        ## 分割
        forimpute = x_train.iloc[train_index].copy()
        imputed = x_train.iloc[valid_index].copy()
        ## 欠損補完用 df
        forimpute = forimpute[columnsall].groupby(columns,as_index=False).count()
        forimpute = forimpute.rename(columns={"index":"count"})
        forimpute = forimpute.sort_values("count",ascending=False)
        forimpute = forimpute.drop_duplicates(groupkey,keep="first")[columns]
        forimpute = forimpute.rename(columns={impute_col:f"{impute_col}_impute"})
        ## not null df
        dfnotnull = imputed[~imputed[impute_col].isnull()].copy()
        dfnotnull[f"{impute_col}_was_null"] = 0
        ## null df
        dfnull = imputed[imputed[impute_col].isnull()].copy()
        dfnull = dfnull.merge(forimpute,on=groupkey,how="left")
        dfnull[impute_col] = dfnull[f"{impute_col}_impute"]
        dfnull = dfnull.drop(f"{impute_col}_impute",axis=1)
        dfnull[f"{impute_col}_was_null"] = 1
        ## concat
        imputed = pd.concat([dfnotnull,dfnull])
        dfreturn = pd.concat([dfreturn,imputed])
    ## modify return df
    dfreturn.index = list(dfreturn["index"])
    dfreturn = dfreturn.drop(["index"],axis=1)
    dfreturn = dfreturn.sort_index()
    return dfreturn