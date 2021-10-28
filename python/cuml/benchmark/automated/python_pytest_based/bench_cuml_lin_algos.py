import pytest

import pytest_benchmark
# FIXME: Remove this when rapids_pytest_benchmark.gpubenchmark is available
# everywhere
try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
          "falling back to pytest_benchmark fixtures.\n")

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass

import cuml
import rmm
from .params import FIXTURE_PARAMS

########
#Helpers
def createData():
    pass

# Record the current RMM settings so reinitialize() will be called only when a
# change is needed (RMM defaults both values to False). This allows the
# --no-rmm-reinit option to prevent reinitialize() from being called at all
# (see conftest.py for details).
RMM_SETTINGS = {"managed_mem": False,
                "pool_alloc": False}


def reinitRMM(managed_mem, pool_alloc):

    if (managed_mem != RMM_SETTINGS["managed_mem"]) or \
       (pool_alloc != RMM_SETTINGS["pool_alloc"]):

        rmm.reinitialize(
            managed_memory=managed_mem,
            pool_allocator=pool_alloc,
            initial_pool_size=2 << 27
        )
        RMM_SETTINGS.update(managed_mem=managed_mem,
                            pool_alloc=pool_alloc)