#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

NP ?= 1
MPIRUN ?= mpirun

MPICXX=mpicxx
CUML_HOME ?= /home/jkraus/workspace/RAPIDS/cuml
MPICXX_FLAGS = -I$(CUML_HOME)/cuML/src -I$(CUML_HOME)/ml-prims/src -I$(CUML_HOME)/comms/include -I$(CUDA_HOME)/include -std=c++11
LD_FLAGS = -L$(CUML_HOME)/cuML/build -lcuml++ -L$(CUML_HOME)/comms/build -lcumlcomms

simple_mpi: Makefile simple_mpi.cpp 
	$(MPICXX) $(MPICXX_FLAGS) simple_mpi.cpp $(LD_FLAGS) -o simple_mpi

.PHONY.: clean
clean:
	rm -f simple_mpi

run: simple_mpi
	$(MPIRUN) -np $(NP) ./simple_mpi
