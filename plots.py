import matplotlib.pyplot as plt

def boxplot_dataframe(
    df,
    tcol,
    gcol,
    nastr="value is N/A",
    vmin=0,
    vmax=0,
    outlier="",
    width=0,
    height=0,
    grid=False
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