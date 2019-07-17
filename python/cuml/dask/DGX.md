# Using IB for Dask + cuML MG on a DGX

## 1. Install UCX from source

```
./autogen
mkdir build && cd build
../configure --prefix=$CONDA_PREFIX --with-cuda=/usr/local/cuda --enable-mt --disable-cma CPPFLAGS="-I//usr/local/cuda/include"
make -j install
```

Verify with `ucx_info -d`

## 2. Set up ucx-py

```
git clone git@github.com:rapidsai/ucx-py
cd ucx-py

export UCX_PATH=$CONDA_PREFIX
make -j install
```


## 3. Enable IB interface at ib0 from [this link](https://docs.oracle.com/cd/E19436-01/820-3522-10/ch4-linux.html#50536461_82843)

*NOTE:* IP over IB kernel module should be enabled. If it is not, you will need to install the driver for it using modprobe

```
sudo ifconfig ib0 10.0.0.50/24
```

Verify interface: `ifconfig ib0`

## 4.  Set UCX environment vars

Use `ibstatus` to see your open IB devices and ports and put them in `UCX_NET_DEVICES` var:

```
export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_2:1,mlx5_1:1
```

Set transports for UCX to use:
```
export UCX_TLS=rc,cuda_copy
```

## 5. Start Dask cluster on ib0 interface:

Run this on a node designated for the scheduler:
```
dask-scheduler --protocol ucx --interface ib0 --scheduler-file=cluster.json
```

And run this on each worker:
```
dask-cuda-worker --scheduler-file cluster.json
```

## 6. Run cumlCommunicator test:

```
from dask.distributed import Client
from cuml.dask.common.comms import CommsContext
from cuml.dask.common import worker_state
from cuml.dask.common import perform_test_comms_send_recv

import random

c = Client("ucx://10.0.0.50:8786")
cb = CommsContext(comms_p2p=True)
cb.init()

n_trials = 10000

def func_test_send_recv(sessionId, n_trials, r):
    handle = worker_state(sessionId)["handle"]
    return perform_test_comms_send_recv(handle, n_trials)

dfs=[c.submit(func_test_send_recv, cb.sessionId, n_trials, random.random(), workers=[w]) for wid, w in zip(range(len(cb.worker_addresses)), cb.worker_addresses)]
dfs
```

You should the following output on your workers:

```

=========================
Trial 0
Rank 0 received: [1, 4, 5, 2, 7, 6, 3]
Rank 1 received: [4, 0, 5, 2, 7, 6, 3]
Rank 2 received: [4, 1, 5, 0, 7, 6, 3]
Rank 3 received: [4, 1, 0, 2, 5, 6, 7]
Rank 4 received: [0, 1, 2, 5, 7, 6, 3]
Rank 5 received: [4, 1, 0, 7, 2, 6, 3]
Rank 6 received: [4, 1, 0, 5, 2, 7, 3]
Rank 7 received: [4, 1, 0, 5, 2, 3, 6]
=========================
=========================
Trial 1
Rank 0 received: [3, 4, 5, 6, 1, 2, 7]
Rank 1 received: [4, 6, 3, 0, 5, 2, 7]
Rank 2 received: [4, 3, 5, 6, 1, 0, 7]
Rank 3 received: [4, 5, 6, 0, 1, 2, 7]
Rank 4 received: [6, 3, 0, 5, 1, 2, 7]
Rank 5 received: [4, 3, 6, 1, 0, 2, 7]
Rank 6 received: [4, 3, 0, 5, 2, 1, 7]
Rank 7 received: [4, 3, 6, 5, 0, 1, 2]
=========================
```


