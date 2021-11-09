from cuml.benchmark import datagen, algorithms
import dask.array as da


def to_dask_array(np_array, client):
    if np_array is not None:
        n_rows = np_array.shape[0]
        n_workers = len(client.scheduler_info()['workers'])
        dask_array = da.from_array(np_array, chunks=n_rows // n_workers)
        return dask_array


def _benchmark_algo(
    benchmark,
    name,
    dataset_name,
    n_samples=10000,
    n_features=100,
    input_type='numpy',
    data_kwargs={},
    algo_args={},
    client=None
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

    if client:
        algo_args['client'] = client
        data = [to_dask_array(d, client) for d in data]

    def _benchmark_inner():
        algo.run_cuml(data, **algo_args)

    benchmark(_benchmark_inner)
