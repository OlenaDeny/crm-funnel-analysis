"""
Shared helper functions for the CRM analytics project.

Used across the data-preparation notebooks (calls, contacts, deals, spend)
and the analytics notebooks (product analytics, final analytics).

Note: this module originally contained several exploratory helper functions
(pair_cat, catboost_encode, plot_pdf_cdf_with_threshold,
plot_pdf_cdf_two_thresholds, visualize_two_sample_test,
visualize_paired_test) that were written during development but never used
in the final notebooks. They were removed here to keep the module focused
on what the project actually runs.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from IPython.display import display

COLOR_TEXT = plt.get_cmap('PuBu')(0.85)  # color for subtitles
FIG_WIDTH = 10
FIG_HEIGHT = 5

# *******************************************************************************************************************
# Quick reload in notebooks (targeted, after updates to helpers.py)

# import importlib
# import helpers as h
# importlib.reload(h)
# *******************************************************************************************************************


def descr_df(df, include='number', show=True, show_stats=True, show_sample_rows=False):
    """
    Prints key information about a dataset as a convenient summary table.

    Builds a summary table with per-column info: dtype, value count,
    missing values, unique values, and descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to analyze

    include : str or list, default='number'
        Data type filter for columns (passed to select_dtypes):
        - 'number' - numeric columns only
        - 'object' - text/categorical columns only
        - ['number', 'object'] - all types

    show : bool, default=True
        If True, prints the table.
        If False, returns the resulting DataFrame instead.

    show_stats : bool, default=True
        Whether to show descriptive statistics (min, mean, median, max).
        Applies to numeric columns only.

    show_sample_rows : bool, default=False
        Whether to show sample values from the first rows of the dataset.
        Useful for a quick content check.

    Returns
    -------
    None or pd.DataFrame
        If show=True, prints the table and returns None.
        If show=False, returns the info DataFrame.

    Examples
    --------
    >>> # Basic info for numeric columns
    >>> descr_df(data)

    >>> # Info for text columns without statistics
    >>> descr_df(data, include='object', show_stats=False)

    >>> # Full info for all columns, with sample values
    >>> descr_df(data, include=['number', 'object'], show_sample_rows=True)
    """

    # Filter columns by data type
    filtered_df = df.select_dtypes(include=include).copy()

    if filtered_df.empty:
        print(f"No columns of type {include} found in the dataset")
        return None

    # Build a dict with basic info
    info_dict = {}
    info_dict['Column name'] = filtered_df.columns
    info_dict['Dtype'] = filtered_df.dtypes.values
    info_dict['Value count'] = filtered_df.count().values
    info_dict['Missing (NaN)'] = filtered_df.isnull().sum().values
    info_dict['Unique values'] = filtered_df.nunique(dropna=True).values

    # Add sample values from the first rows (optional)
    if show_sample_rows:
        SAMPLE_ROWS_COUNT = 3
        for i in range(min(SAMPLE_ROWS_COUNT, len(filtered_df))):
            info_dict[f'Sample row {i+1}'] = filtered_df.iloc[i].values

    # Add statistics for numeric columns (optional)
    if show_stats:
        numeric_df = filtered_df.select_dtypes(include='number')

        if not numeric_df.empty:
            # Use reindex to align correctly with all columns
            info_dict['Min'] = numeric_df.min().reindex(filtered_df.columns).values
            info_dict['Q1'] = numeric_df.quantile(0.25).reindex(filtered_df.columns).values
            info_dict['Mean'] = numeric_df.mean().reindex(filtered_df.columns).values
            info_dict['Q2 Median'] = numeric_df.median().reindex(filtered_df.columns).values
            info_dict['Q3'] = numeric_df.quantile(0.75).reindex(filtered_df.columns).values
            info_dict['Max'] = numeric_df.max().reindex(filtered_df.columns).values
            info_dict['Range'] = numeric_df.max().reindex(filtered_df.columns).values - numeric_df.min().reindex(filtered_df.columns).values
            info_dict['IQR'] = numeric_df.quantile(0.75).reindex(filtered_df.columns).values - numeric_df.quantile(0.25).reindex(filtered_df.columns).values

    # Build the final DataFrame
    result_df = pd.DataFrame(info_dict)

    # Print or return the result
    if show:
        display(result_df)
        return None
    else:
        return result_df

# *******************************************************************************************************************


def hist_box(column, df, title=None, discrete=False, bins='fd', hue=None, figsize=(FIG_WIDTH, FIG_HEIGHT), kde=False, p=1):
    """
    Visualizes the distribution of a numeric variable with a histogram and a boxplot.

    Builds a combined figure:
    - Top: histogram with an optional density curve (KDE)
    - Bottom: boxplot for spotting outliers

    Parameters
    ----------
    column : str
        Name of the numeric column to analyze

    df : pd.DataFrame
        Dataframe with the data

    title : str, optional
        Plot title. Defaults to the column name if not given.

    bins : int or str, default='fd'
        Number of histogram bins, or the method used to compute them:
        - 'fd' (Freedman-Diaconis) - automatic choice (recommended)
        - 'auto', 'sturges', 'scott' - other automatic methods
        - int (e.g. 30, 50) - exact number of bins

    hue : str, optional
        Categorical column to group data by.
        Lets you compare distributions across groups on one plot.

    figsize : tuple, default=(FIG_WIDTH, FIG_HEIGHT)
        Figure size (width, height) in inches.

    Examples
    --------
    >>> # Simple price distribution
    >>> hist_box('Price', data, title='Car price')

    >>> # Compare price distribution by fuel type
    >>> hist_box('Price', data, title='Price by fuel type', hue='FuelType')

    >>> # With a fixed number of bins
    >>> hist_box('Mileage', data, bins=50, title='Mileage')
    """

    # Create the figure and grid layout
    fig = plt.figure(figsize=figsize)
    n = 5  # rows reserved for the histogram
    p = p  # rows for the boxplot
    gs = GridSpec(ncols=1, nrows=n+p, figure=fig, hspace=0.05)

    # Default title
    if title is None:
        title = column

    # === HISTOGRAM (top) ===
    ax_hist = fig.add_subplot(gs[0:n, 0])

    if hue is None:
        # Simple histogram, no grouping
        sns.histplot(
            data=df,
            x=column,
            bins=bins,
            kde=kde,
            ax=ax_hist,
            alpha=0.6,
            discrete=discrete
        )
    else:
        # Histogram grouped by category
        sns.histplot(
            data=df,
            x=column,
            hue=hue,
            bins=bins,
            kde=True,
            ax=ax_hist,
            alpha=0.5
        )
        # Grab the legend seaborn created and set its title
        legend = ax_hist.get_legend()
        if legend:
            legend.set_title(hue)

    ax_hist.set_xlabel("")
    ax_hist.set_ylabel('Frequency (count)', fontsize=13)
    ax_hist.tick_params(axis='x', labelbottom=False)
    ax_hist.tick_params(axis='y', labelsize=12)
    ax_hist.grid(axis='y', alpha=0.3, linestyle='--')

    # === BOXPLOT (bottom) ===
    ax_box = fig.add_subplot(gs[n:n+p, 0])

    if hue is None:
        # Simple boxplot
        sns.boxplot(
            data=df,
            x=column,
            ax=ax_box,
            width=0.3
        )
    else:
        # Boxplot grouped and colored by category
        sns.boxplot(
            data=df,
            x=column,
            y=hue,
            hue=hue,
            ax=ax_box,
            legend=False,
        )

    ax_box.set_xlabel(column, fontsize=13)
    ax_box.tick_params(axis='both', labelsize=12)
    ax_box.grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle(f'Histogram and boxplot for "{title}"', fontsize=16, color=COLOR_TEXT)
    plt.subplots_adjust(top=0.92)
    plt.show()

# *******************************************************************************************************************


def iqr_outliers(column, df, coef=1.5, show=True):
    """
    Finds the bounds and count of outliers using the interquartile range (IQR) method.

    IQR method:
    - Computes the 25th (Q1) and 75th (Q3) percentiles
    - IQR = Q3 - Q1
    - Lower bound: Q1 - coef * IQR
    - Upper bound: Q3 + coef * IQR
    - Values outside these bounds are considered outliers

    Parameters
    ----------
    column : str
        Name of the numeric column to analyze

    df : pd.DataFrame
        Dataframe with the data

    show : bool, default=True
        If True, prints the result table.
        If False, returns the resulting DataFrame instead.

    Returns
    -------
    None or pd.DataFrame
        If show=True, prints the table and returns None.
        If show=False, returns a DataFrame describing the outliers.
    """
    series = df[column].dropna().copy()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    left_border = q1 - coef * iqr
    right_border = q3 + coef * iqr

    outlier_count_left = (series < left_border).sum()
    outlier_count_right = (series > right_border).sum()
    total = len(series)

    result_dict = {
        'Outlier bounds': [left_border, right_border],
        'Outlier count': [outlier_count_left, outlier_count_right],
        'Outlier %': [round(outlier_count_left / total * 100, 2), round(outlier_count_right / total * 100, 2)]
    }

    outlier_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Left', 'Right'])

    if show:
        display(outlier_df)
    else:
        return outlier_df

# *******************************************************************************************************************


def colls_stats(df, cols):
    """
    Comparison table of column statistics, useful before/after a transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to analyze

    cols : list[str]
        Column names to analyze

    Returns
    -------
    pd.DataFrame with, per column:
        name, dtype, value count, missing count, missing %,
        non-missing count, non-missing %, unique count

    Example
    -------
    >>> cols = ['created_date', 'sla']
    >>> colls_stats(df, cols)

    Recommended usage: run once before a transformation and once after,
    to compare the two tables.
    """
    rows = []
    for col in cols:
        rows.append({
            'Column name': col,
            'Dtype': df[col].dtype,
            'Value count': df[col].count(),
            'Missing (NaN)': df[col].isna().sum(),
            'Missing %': round(df[col].isna().mean() * 100, 2),
            'Non-missing': df[col].notna().sum(),
            'Non-missing %': round(df[col].notna().mean() * 100, 2),
            'Unique': df[col].nunique()
        })
    return pd.DataFrame(rows)

# *******************************************************************************************************************


def clean_marketing_hierarchy(dataframe, name="Dataset"):
    """
    Generic cleanup of the marketing hierarchy (source -> campaign -> ad_group -> ad).
    Works for both Deals and Spend.

    # --- USAGE IN THE PROJECT ---

    # 1. In the deals notebook
    # deals = pd.read_pickle(PROCESSED / 'deals_cleaned.pkl')
    deals_cleaned = clean_marketing_hierarchy(df, "DEALS")

    # 2. In the spend notebook
    # spend = pd.read_pickle(PROCESSED / 'spend_cleaned.pkl')
    spend_cleaned = clean_marketing_hierarchy(spend, "SPEND")
    """
    # 1. Work on a copy so we don't accidentally modify the original
    df = dataframe.copy()

    # Fields we expect to find
    cols = ['source', 'campaign', 'ad_group', 'ad']

    print(f"=== Processing hierarchy: {name} ===")

    # 2. Technical cleanup: remove junk values that break groupby operations
    for col in cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.strip()
                       .replace(['nan', 'NaN', 'None', '', '-', 'None', 'nan'], np.nan))

    # 3. Recovery logic (only where it's safe to do so)
    # We only fill in a value if the parent has a SINGLE distinct child value.

    hierarchy = [
        ('ad', 'ad_group'),
        ('ad_group', 'campaign'),
        ('campaign', 'source')
    ]

    for child, parent in hierarchy:
        if child in df.columns and parent in df.columns:
            n_before = df[child].isna().sum()

            # Approach:
            # If a parent has only one distinct child value, fill missing values with it.
            # If there are multiple candidate values, leave it as NaN rather than guess.
            df[child] = df.groupby(parent)[child].transform(
                lambda x: x.fillna(x.dropna().iloc[0]) if x.dropna().nunique() == 1 else x
            )

            n_after = df[child].isna().sum()
            print(f"  [{child}] by [{parent}]: recovered {n_before - n_after} values")

    # 4. Final gap-filling (important for downstream groupby/merge operations)
    # Replace any remaining NaNs with an explicit label
    for col in cols:
        if col in df.columns:
            filler = f"(no {col})"
            df[col] = df[col].fillna(filler)

    print(f"--- Done for {name}. Remaining missing values: 0 ---\n")
    return df
