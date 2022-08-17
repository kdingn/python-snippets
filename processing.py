import numpy as np

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
    """
    uniques = list(df[col].unique())
    uniques.remove(np.nan)
    allel = []
    for el in list(map(lambda x : x.split(", "), uniques)):
        allel.extend(el)
    return list(set(allel))