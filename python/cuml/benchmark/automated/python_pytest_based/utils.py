from cuml.benchmark import datagen, algorithms

def _benchmark_algo(
    benchmark,
    name,
    dataset_name,
    n_samples=10000,
    n_features=100,
    input_type='numpy',
    data_kwargs={},
    algo_args={},
):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset
    'dataset_name'"""
    algo = algorithms.algorithm_by_name(name)
    data = datagen.gen_data(
        dataset_name,
        input_type,
        n_samples=n_samples,
        n_features=n_features,
        **data_kwargs
    )

    def _benchmark_inner():
        algo.run_cuml(data, **algo_args)

    benchmark(_benchmark_inner)