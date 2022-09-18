import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_box_cat(
    df,
    tcol,
    gcol,
    nastr="value is N/A",
    vmin=0,
    vmax=0,
    outlier="",
    width=0,
    height=0,
    grid=False,
    sort_labels=False
):
    """
    Output values as a boxplot from pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for boxplot.
    tcol : str
        Target column name. Target is supposed to be float.
    gcol : str
        Column for group-by key. Group-key is supposed to be strings.
    nastr : str
        Key name for N/A value.
    vmin : float
        Min value for boxplot.
    vmax : float
        Max value for boxplot.
    outlier : str
        Outlier format.
    width : float
        Width of plot.
    height : float
        Height of plot.
    grid : bool
        If True, grid lines appear.
    sort_labels : bool
        If True, labels are sorted.
    """
    # formating dataframe
    df = df[[gcol,tcol]]
    df = df.fillna(nastr)

    # labels
    elabels = list(df.groupby(gcol,as_index=False).median().sort_values(tcol)[gcol])
    if nastr in elabels:
        elabels.remove(nastr)
        labels = [nastr]
    else:
        labels = []
    if sort_labels:
        elabels.sort()
    labels.extend(elabels)
    
    # boxplot
    if width*height==0:
        fig = plt.figure()
    else:
        fig = plt.figure(
            figsize=[width,height]
        )
    ax = fig.add_subplot()
    ax.boxplot(
        [df[df[gcol]==label][tcol].values for label in labels],
        labels=labels,
        vert=False,
        showmeans=True,
        sym=outlier
    )
    if vmax!=vmin:
        ax.set_xlim(vmin,vmax)
    ax.set_xlabel(tcol)
    ax.set_ylabel(gcol)
    ax.grid(grid)
    plt.show()


def plot_boxbar_cat(
    df,
    tcol,
    gcol,
    nastr="value is N/A",
    vmin=0,
    vmax=0,
    outlier="",
    width=0,
    height=0,
    wspace=0.05,
    grid_left=False,
    grid_right=False,
    ratio=0.8,
    sort_labels=False
):
    """
    Output values as boxplot and barplot from pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for boxplot.
    tcol : str
        Target column name. The value is supposed to be float and non-null.
    gcol : str
        Column for group-by key. The value is supposed to be strings.
    nastr : str
        Key name for N/A value.
    vmin : float
        Min value for boxplot.
    vmax : float
        Max value for boxplot.
    outlier : str
        Outlier format.
    width : float
        Width of plot.
    height : float
        Height of plot.
    wspace : float
        Space for the plots.
    grid_left : bool
        If True, grid lines of the left plot appear.
    grid_right : bool
        If True, grid lines of the right plot appear.
    ratio : float, 0~1
        Ratio where the left plot occupying.
    sort_labels : bool
        If True, labels are sorted.
    """
    # formating dataframe
    df = df[[gcol,tcol]]
    df = df.fillna(nastr)

    # labels
    elabels = list(df.groupby(gcol,as_index=False).median().sort_values(tcol)[gcol])
    if nastr in elabels:
        elabels.remove(nastr)
        labels = [nastr]
    else:
        labels = []
    if sort_labels:
        elabels.sort()
    labels.extend(elabels)
    
    # plot
    if width*height==0:
        fig, ax = plt.subplots(
            1, 2, 
            gridspec_kw={
                'width_ratios':[ratio, 1-ratio],
                "wspace":wspace
            },
        )
    else:
        fig, ax = plt.subplots(
            1, 2, 
            gridspec_kw={
                'width_ratios':[ratio, 1-ratio],
                "wspace":wspace
            },
            figsize=(width,height)
        )
    # boxplot
    ax[0].boxplot(
        [df[df[gcol]==label][tcol].values for label in labels],
        labels=labels,
        vert=False,
        showmeans=True,
        sym=outlier
    )
    if vmax!=vmin:
        ax[0].set_xlim(vmin,vmax)
    ax[0].set_xlabel(tcol)
    ax[0].set_ylabel(gcol)
    ax[0].grid(grid_left)
    # barplot
    ax[1].barh(
        labels,
        [len(df[df[gcol]==label]) for label in labels]
    )
    ax[1].set_xlabel("row counts")
    ax[1].tick_params(labelleft=False)
    ax[1].grid(grid_right)
    plt.show()


def plot_box_num(
    df,
    tcol,
    gcol,
    nastr="value is N/A",
    vmin=0,
    vmax=0,
    outlier="",
    width=0,
    height=0,
    grid=False,
    sort_labels=True,
    bins=10
):
    """
    Output values as a boxplot from pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for boxplot.
    tcol : str
        Target column name. Target is supposed to be float.
    gcol : str
        Column for group-by key. Group-key is supposed to be strings.
    nastr : str
        Key name for N/A value.
    vmin : float
        Min value for boxplot.
    vmax : float
        Max value for boxplot.
    outlier : str
        Outlier format.
    width : float
        Width of plot.
    height : float
        Height of plot.
    grid : bool
        If True, grid lines appear.
    sort_labels : bool
        If True, labels are sorted.
    bins : int
        The number of bins for quantile cutting.
    """
    # formating dataframe
    df = df[[gcol,tcol]].copy()
    
    # creating bins from numeric variable
    ccol = gcol + "_bin"
    df[ccol] = pd.qcut(df[gcol],bins).astype(str)

    # sort bins
    uniques = df[ccol].unique()
    dfsort = pd.DataFrame({
        ccol:uniques,
        f"lower_boundary":list(map(lambda x : x[1:].split(",")[0], uniques))
    })
    dfsort["lower_boundary"] = \
    dfsort["lower_boundary"].replace(
        "an",
        str(float(dfsort["lower_boundary"].min())-1)
    ).apply(lambda x : float(x))
    dfsort = dfsort.sort_values("lower_boundary").reset_index(drop=True)
    dfsort["bins"] = pd.Series(dfsort.index).apply(lambda x : "bin"+str(x).zfill(2))
    df = df.merge(dfsort[[ccol,"bins"]],on=ccol)
    df[ccol] = df["bins"] + "_" + df[ccol].replace("nan",np.nan)

    # plot
    plot_box_cat(
        df,
        tcol,
        ccol,
        nastr=nastr,
        vmin=vmin,
        vmax=vmax,
        outlier=outlier,
        width=width,
        height=height,
        grid=grid,
        sort_labels=sort_labels
    )


def plot_boxbar_num(
    df,
    tcol,
    gcol,
    nastr="value is N/A",
    vmin=0,
    vmax=0,
    outlier="",
    width=0,
    height=0,
    wspace=0.05,
    grid_left=False,
    grid_right=False,
    ratio=0.8,
    sort_labels=True,
    bins=10
):
    """
    Output values as boxplot and barplot from pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe for boxplot.
    tcol : str
        Target column name. The value is supposed to be float and non-null.
    gcol : numeric
        Column for group-by key. The value is supposed to be numeric.
    nastr : str
        Key name for N/A value.
    vmin : float
        Min value for boxplot.
    vmax : float
        Max value for boxplot.
    outlier : str
        Outlier format.
    width : float
        Width of plot.
    height : float
        Height of plot.
    wspace : float
        Space for the plots.
    grid_left : bool
        If True, grid lines of the left plot appear.
    grid_right : bool
        If True, grid lines of the right plot appear.
    ratio : float, 0~1
        Ratio where the left plot occupying.
    sort_labels : bool
        If True, labels are sorted.
    bins : int
        The number of bins for quantile cutting.
    """
    # formating dataframe
    df = df[[gcol,tcol]].copy()
    
    # creating bins from numeric variable
    ccol = gcol + "_bin"
    df[ccol] = pd.qcut(df[gcol],bins,duplicates="drop").astype(str)

    # sort bins
    uniques = df[ccol].unique()
    dfsort = pd.DataFrame({
        ccol:uniques,
        f"lower_boundary":list(map(lambda x : x[1:].split(",")[0], uniques))
    })
    dfsort["lower_boundary"] = \
    dfsort["lower_boundary"].replace(
        "an",
        str(float(dfsort["lower_boundary"].min())-1)
    ).apply(lambda x : float(x))
    dfsort = dfsort.sort_values("lower_boundary").reset_index(drop=True)
    dfsort["bins"] = pd.Series(dfsort.index).apply(lambda x : "bin"+str(x).zfill(2))
    df = df.merge(dfsort[[ccol,"bins"]],on=ccol)
    df[ccol] = df["bins"] + "_" + df[ccol].replace("nan",np.nan)

    # plot
    plot_boxbar_cat(
        df,
        tcol,
        ccol,
        nastr=nastr,
        vmin=vmin,
        vmax=vmax,
        outlier=outlier,
        width=width,
        height=height,
        wspace=wspace,
        grid_left=grid_left,
        grid_right=grid_right,
        ratio=ratio,
        sort_labels=sort_labels
    )
