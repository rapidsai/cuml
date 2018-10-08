 # Copyright (c) 2018, NVIDIA CORPORATION.
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

import numpy as np

def write_mortgage_script(output='test_mortgage.sh'):
    params = {
        'python':['test_pca.py','test_tsvd.py'],
        'data':['mortgage'],
        'nrows':np.power(2,np.arange(18,24)),
        'ncols':np.power(2,np.arange(7,10)),
        'random_state':[16,42,100],
        'quarters':[16],
        'use_assert':[0],
        'threshold':[1e-3],
        'test_model':['cuml']
        #'test_model':['h2o4gpu']
    }
    cols = ['python','data','nrows','ncols',
        'random_state','quarters','use_assert','threshold',
        'test_model',
    ]
    assert len(cols)==len(params)
    write_script(output,params,cols)

def write_script(output,params,cols):
    with open(output,'w') as f:
        pass
    permutation([],0,cols,params,output)    

def permutation(pre,step,cols,params,output):
    if step == len(cols):
        print(pre)
        line = ' '.join(pre)
        with open(output,'a') as f:
            f.write('%s\n'%line)
        return
    if len(pre)==step:
        pre.append('')
    for v in params[cols[step]]:
        if cols[step] == 'python':
            pre[step] = '%s %s'%(cols[step],v)
        else:
            pre[step] = '--%s %s'%(cols[step],v)
        permutation(pre,step+1,cols,params,output)

if __name__ == '__main__':
    write_mortgage_script(output='test_mortgage.sh') 
