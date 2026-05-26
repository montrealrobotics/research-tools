"""
sudo pip3 install pandas
sudo pip3 install seaborn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cleanrl', 'plotting'))
from make_plots import render_plot, colors, linestyle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

# Extend the shared dicts with biped-specific labels
colors.update({
    'SMiRL (ours)': 'k',
    'SMiRL VAE (ours)': 'purple',
    'ICM': 'b',
    'RND': 'orange',
    'Oracle': 'g',
    'Reward + SMiRL (ours)': 'k',
    'Reward + ICM': 'b',
    'Reward': 'r',
    'SMiRL + ICM': 'brown',
})
linestyle.update({
    'SMiRL (ours)': '-',
    'SMiRL VAE (ours)': '--',
    'ICM': '-',
    'RND': '--',
    'Oracle': '--',
    'Reward + SMiRL (ours)': '-',
    'Reward + ICM': '-',
    'Reward': '--',
    'SMiRL + ICM': '-',
})


def get_biped_data_frame(df, row_indices, col_name, res=5, step_size=20000, invert=False):
    """Load and smooth rows from the biped CSV into a tidy DataFrame.

    :param df:           pandas DataFrame loaded from safe.csv
    :param row_indices:  list of row indices (df.iloc[i]) to concatenate as seeds
    :param col_name:     name to give the value column
    :param res:          smoothing window size
    :param step_size:    env steps between each data point
    :param invert:       if True, compute (1 - value) before smoothing (for fall rates)
    """
    data = []
    for row_idx in row_indices:
        bf = np.array([float(x) for x in list(df.iloc[row_idx][1:])])
        if invert:
            bf = 1.0 - bf
        bf = (np.cumsum(bf)[res:] - np.cumsum(bf)[:-res]) / res
        time = step_size
        for val in bf:
            data.append((time, float(val)))
            time += step_size
    result = pd.DataFrame(data)
    result = result.rename(columns={0: 'Steps', 1: col_name})
    return result


def add_biped_plot(ax, df, row_indices, col_name, label, res=5, step_size=20000, invert=False):
    """Plot one condition from the biped CSV onto *ax*.

    :param ax:          matplotlib Axes
    :param df:          pandas DataFrame loaded from safe.csv
    :param row_indices: list of row indices to use as seeds
    :param col_name:    intermediate column name for the tidy DataFrame
    :param label:       legend label (must be a key in colors/linestyle)
    :param res:         smoothing window
    :param step_size:   env steps between data points
    :param invert:      compute (1 - value) first, for fall-rate data
    """
    plot_data = get_biped_data_frame(df, row_indices, col_name, res=res,
                                     step_size=step_size, invert=invert)
    sns.lineplot(data=plot_data, x='Steps', y=col_name, ax=ax,
                 label=label, c=colors[label])
    ax.lines[-1].set_linestyle(linestyle[label])


if __name__ == '__main__':

    df = pd.read_csv('./safe.csv')
    res = 5

    # -------------------------------------------------------------------------
    # Figure 0 — Walk task: % Episodes w/ Falls
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(20 / 3., 5))

    add_biped_plot(ax, df, row_indices=[15, 16],        col_name='Biped Falls',
                   label='Reward + SMiRL (ours)', res=res, invert=True)
    add_biped_plot(ax, df, row_indices=[20, 21, 22],    col_name='Biped Falls',
                   label='Reward + ICM',           res=res, invert=True)
    add_biped_plot(ax, df, row_indices=[11, 12],        col_name='Biped Falls',
                   label='Reward',                 res=res, invert=True)

    render_plot(ax, fig, title='Walk task: % Episodes w/ Falls',
                ylabel='% Episodes w/ Falls', xlabel='Env Steps', outpath='file0')

    # -------------------------------------------------------------------------
    # Figure 1 — Walk task: Walk Reward
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(20 / 3., 5))

    add_biped_plot(ax, df, row_indices=[25, 26, 27],    col_name='Biped Reward + SMiRL',
                   label='Reward + SMiRL (ours)', res=res, invert=False)
    add_biped_plot(ax, df, row_indices=[37, 38, 39],    col_name='Biped Reward + ICM',
                   label='Reward + ICM',           res=res, invert=False)
    add_biped_plot(ax, df, row_indices=[29, 30, 31],    col_name='Biped Reward',
                   label='Reward',                 res=res, invert=False)

    ax.set_xlim(0, 2e6)
    render_plot(ax, fig, title='Walk task: Walk Reward',
                ylabel='r_walk', xlabel='Env Steps', outpath='file1')
