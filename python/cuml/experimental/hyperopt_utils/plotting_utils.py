#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import seaborn as sns
import matplotlib.pyplot as plt
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")


def plot_heatmap(df, col1, col2):
    """
    Generates a heatmap to highlight interactions of two parameters specified
    in col1 and col2.

    Parameters
    ----------
    df : Pandas dataframe
         Results from Grid or Random Search
    col1 : string; Name of the first parameter
    col2: string; Name of the second parameter

    """
    max_scores = df.groupby([col1, col2]).max()
    max_scores = max_scores.unstack()[["mean_test_score"]]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt=".3g")


def plot_search_results(res):
    """
    Plots by fixing all parameters except one parameter to its best value using
    matplotlib.

    Accepts results from grid or random search from dask-ml.

    Parameters
    ----------
    res : results from Grid or Random Search

    """
    # Results from grid search
    results = res.cv_results_
    means_test = results["mean_test_score"]
    stds_test = results["std_test_score"]
    # Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(res.best_params_.keys())
    for p_k, p_v in res.best_params_.items():
        masks.append(list(results["param_" + p_k].data == p_v))
    try:
        # Grid Search
        params = res.param_grid
        # Plotting results
        fig, ax = plt.subplots(
            1, len(params), sharex="none", sharey="all", figsize=(20, 5)
        )
        fig.suptitle("Score per parameter")
        fig.text(0.04, 0.5, "MEAN SCORE", va="center", rotation="vertical")
        pram_preformace_in_best = {}
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i + 1 :])
            pram_preformace_in_best
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            x = np.array(params[p])
            y_1 = np.array(means_test[best_index])
            e_1 = np.array(stds_test[best_index])
            ax[i].errorbar(
                x, y_1, e_1, linestyle="--", marker="o", label="test"
            )
            ax[i].set_xlabel(p.upper())
    except Exception as e:
        # Randomized Search
        print("Cannot generate plots because of ", type(e), "trying again...")
        try:
            params = res.param_distributions
            # Plotting results
            fig, ax = plt.subplots(
                1, len(params), sharex="none", sharey="all", figsize=(20, 5)
            )
            fig.suptitle("Score per parameter")
            fig.text(0.04, 0.5, "MEAN SCORE", va="center", rotation="vertical")

            for i, p in enumerate(masks_names):
                results = pd.DataFrame(res.cv_results_)
                select_names = masks_names[:i] + masks_names[i + 1 :]
                for j in select_names:
                    best_value = res.best_params_[j]
                    results = results[results["param_" + j] == best_value]

                x = np.array(results["param_" + p])
                y_1 = np.array(results["mean_test_score"])
                e_1 = np.array(results["std_test_score"])
                ax[i].errorbar(
                    x, y_1, e_1, linestyle="--", marker="o", label="test"
                )
                ax[i].set_xlabel(p.upper())
        except Exception as e:
            # Something else broke while attempting to plot
            print("Cannot generate plots because of ", type(e))
            return
    plt.legend()
    plt.show()
