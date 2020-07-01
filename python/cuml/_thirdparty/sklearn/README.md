# GPU accelerated Scikit-Learn preprocessing

This directory contains code originating from the Scikit-Learn library. The Scikit-Learn license applies accordingly (see `LICENSE`).

This work is dedicated to providing GPU accelerated tools for preprocessing. The Scikit-Learn code is slightly modified to make it possible to take common inputs used throughout cuML such as Numpy and Cupy arrays, Pandas and cuDF dataframes and compute the results on GPU.

The code originates from the Scikit-Learn Github repository : https://github.com/scikit-learn/scikit-learn.git and is based on version/branch 0.23.1.