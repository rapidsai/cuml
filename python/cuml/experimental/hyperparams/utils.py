import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

def plot_heatmap(df_gridsearch, col1, col2):
    max_scores = df_gridsearch.groupby([col1, col2]).max()
    max_scores = max_scores.unstack()[['mean_test_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.3g')


def plot_search_results(grid):
    """
    Plots by fixing all paramters to their best value
    except the one we are plotting.
    Params:
        grid: A trained GridSearchCV object.
    """
    # Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    # Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))
    try:
        params = grid.param_grid
        # Ploting results
        fig, ax = plt.subplots(1, len(params), sharex='none',
                               sharey='all', figsize=(20,5))
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
            ax[i].errorbar(x, y_1, e_1, linestyle='--',
                           marker='o', label='test')
            ax[i].set_xlabel(p.upper())
    except Exception as e:
        try:
            params = grid.param_distributions
            # Ploting results
            fig, ax = plt.subplots(1, len(params), sharex='none',
                                   sharey='all', figsize=(20,5))
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
                y_1 = np.array(results['mean_test_score'])
                e_1 = np.array(results['std_test_score'])
                ax[i].errorbar(x, y_1, e_1, linestyle='--',
                               marker='o', label='test')
                ax[i].set_xlabel(p.upper())
        except Exception as e:
            print("Cannot generate")
            return
    plt.legend()
    plt.show()
