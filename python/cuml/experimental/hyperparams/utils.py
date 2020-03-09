import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from matplotlib import ticker
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot


def plot_parallel_coord(d, cols):
    df = d.copy(deep=False)
    x = [i for i, _ in enumerate(cols)]

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            ax.plot(x, df.loc[idx, cols])
        ax.set_xlim([x[i], x[i+1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])


    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])


    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    plt.title("Parallel Coordinate plot")

    plt.show()

def plot_3d(df_gridsearch, param="alpha", col1="param_learning_rate", col2="param_min_child_weight"):
    """
        Params:
        - param: Which parameter to fix at the best value
        - col1, col2: the parameters to plot to see the accuracy scores perform
    """
    if param=="min_child_weight":
        best_param = 5
        filtered_df = df_gridsearch[df_gridsearch.param_min_child_weight==best_param]

    elif param=="alpha":
        best_param = 0.001
        filtered_df = df_gridsearch[df_gridsearch.param_alpha==best_param]
    elif param=="learning_rate":
        best_param = 0.15
        filtered_df = df_gridsearch[df_gridsearch.param_learning_rate==best_param]
    elif param=="max_depth":
        best_param = 9
        filtered_df = df_gridsearch[df_gridsearch.param_max_depth==best_param]

    data = [
        go.Scatter3d(
            x=filtered_df[col1],
            y=filtered_df[col2],
            z=filtered_df['mean_test_score'],
            mode='markers',
            marker=dict(
                size=filtered_df.mean_fit_time ** (1 / 3),
                color=filtered_df['mean_test_score'],
                opacity=0.99,
                colorscale='Viridis',
                colorbar=dict(title = 'test score'),
                line=dict(color='rgb(140, 140, 170)'),
                showscale=False,

            ),
            hoverinfo='text+name',
            name='Test set',
        )
    ]
    layout = go.Layout(
    margin=dict(
        l=20,
        r=20,
        b=20,
        t=20
    ),
    #     height=600,
    #     width=960,
    scene = dict(
        xaxis = dict(
            title=col1,
        ),
        yaxis = dict(
            title=col2,
            type='log'
        ),
        zaxis = dict(
            title='Accuracy score',
            type='log'
        ),
        camera = dict(
            eye = dict(
                y = 2.089757339892154,
                x = -0.5464711077183096,
                z = 0.14759264478960377,
                )
            ),
    ),

    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
def plot_heatmap(df_gridsearch, col1, col2):
    max_scores = df_gridsearch.groupby([col1, col2]).max()
    max_scores = max_scores.unstack()[['mean_test_score']]

    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.3g')
def plot_search_results(grid):
    """
    Plots by fixing all paramters to their best value except the one we are plotting.
    Params:
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))
    try:
        params=grid.param_grid
        ## Ploting results
        fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
        fig.suptitle('Score per parameter')
        fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
        pram_preformace_in_best = {}
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i+1:])
            pram_preformace_in_best
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            x = np.array(params[p])
            y_1 = np.array(means_test[best_index])
            e_1 = np.array(stds_test[best_index])
            ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
            ax[i].set_xlabel(p.upper())
    except:
        try:
            params=grid.param_distributions
            ## Ploting results
            fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
            fig.suptitle('Score per parameter')
            fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')

            results = pd.DataFrame(grid.cv_results_)
            # print(results.head())
            for i, p in enumerate(masks_names):
                results = pd.DataFrame(grid.cv_results_)
                select_names = masks_names[:i] + masks_names[i+1:]
                for j in select_names:
                    best_value = grid.best_params_[j]
                    results = results[results['param_'+j] == best_value]

                x = np.array(results['param_'+p])
                y_1 = np.array(results['mean_test_score' ])
                e_1 = np.array(results['std_test_score' ])
                ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
                ax[i].set_xlabel(p.upper())
        except:
            print("Cannot generate")
            return
    

    plt.legend()
    plt.show()
