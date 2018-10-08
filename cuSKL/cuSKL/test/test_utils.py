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

import time
import glob
import os
import sys
import multiprocessing
import pandas as pd
import numpy as np
try:
    import pygdf
    from cuML import PCA as cumlPCA
except:
    print("pygdf or cuml is not installed")
import argparse
import warnings
warnings.filterwarnings("ignore")
LOG_PATH = './log'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', default=1000, type=int, help='number of rows of the input matrix. default:1000')
    parser.add_argument('--ncols', default=512, type=int, help='number of cols of the input matrix. default:512')
    parser.add_argument('--data', default='mortgage', type=str, help='input data: random or mortgage. default:mortgage')
    parser.add_argument('--quarters', default=8, type=int, help='number of quarters of the mortgage data. default:8')
    parser.add_argument('--rows_per_quarter', default=100000, type=int, help='number of rows per quarter of the mortgage data. default:100000')
    parser.add_argument('--use_assert', default=1, type=int, help='assert for equality. default:1')
    parser.add_argument('--test_model', default='cuml', type=str, help='model to be tested. default:cuml')
    parser.add_argument('--random_state', default=42, type=int, help='random state. default:42')
    parser.add_argument('--threshold', default=1e-3, type=float, help='margin for comparison. default:1e-3') 
    args = parser.parse_args()
    return args

def timer(method):
    def timed(*args,**kw):
        model = kw.get('model','')
        start = time.time()
        result = method(*args,**kw)
        end = time.time()
        message = "%s %s done in %.5f seconds"%(method.__name__,model,end-start)
        print(message)
        write_log(message)
        return result
    return timed

def write_log(message, logpath=LOG_PATH):
    with open(logpath,'a') as fo:
        fo.write('%s\n'%message)

@timer
def load_mortgage(quarters = 8, path='/raid/mortgage/performance',rows_per_quarter=100000):
    paths = sorted(glob.glob("%s/*.txt"%path))[:quarters]
    inputs = [(path,rows_per_quarter) for path in paths]
    dfs = parallel_run(load_one_quarter,inputs)
    df = pd.concat(dfs,axis=0)#.reset_index()
    df = clean_encode_normalize(df)
    return df.values.astype(np.float32)

def clean_encode_normalize(df):
    cat_cols,num_cols = [],[]
    for col in df.columns:
        if df[col].dtype == 'object':
            cat_cols.append(col)
        else:
            col_min,col_max = df[col].min(),df[col].max()
            df[col] = (df[col].fillna(df[col].median())-col_min)/(col_max-col_min)
    if len(cat_cols):
        df = pd.get_dummies(data=df, columns=cat_cols, dummy_na=True)
    return df.fillna(0.5)

def load_one_quarter(input_):
    path,nrows = input_   
    df = pd.read_csv(path,sep='|',header=None,nrows=nrows)
    group = path.split('/')[-2]
    df.columns = get_cols(group)
    date_ID_cols = [col for col in df.columns if col.endswith('_id') or col.endswith('_date')] 
    df.drop(date_ID_cols,axis=1,inplace=True)
    return df

def parallel_run(func,data,silent=False):
    p = multiprocessing.Pool()
    results = p.imap(func, data)
    num_tasks = len(data)
    while (True):
        completed = results._index
        if silent==0:
            print("\r--- parallel {} completed {:,} out of {:,}".format(func.__name__,completed, num_tasks),end="")
        sys.stdout.flush()
        time.sleep(1)
        if (completed == num_tasks):
            break
    p.close()
    p.join()
    if silent==0:
        print()
    return list(results) 

def get_cols(group):
    if group == 'acquisition':
        return get_acquisition_cols()
    elif group == 'performance':
        return get_performance_cols()
    else:
        print("%s cols not defined"%group)
        assert 0

def get_acquisition_cols():
    return [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
        'relocation_mortgage_indicator'
    ]

def get_performance_cols():
    return [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]

def parallel_run1(func,data):
    # sequential sanity check
    for d in data:
        print(d)
        func(d)

def pd2pygdf(df):
    if isinstance(df,np.ndarray):
        return np2pygdf(df)
    pdf = pygdf.DataFrame()
    for c,column in enumerate(df):
        pdf[c] = df[column]
    return pdf

def np2pygdf(df):
    pdf = pygdf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:,c]
    return pdf

def test_pygdf(nrows=1000,ncols=1000):
    x = np.random.rand(nrows,ncols).astype(np.float32) 
    df = pd.DataFrame({'fea%d'%i:x[:,i] for i in range(x.shape[1])})
    pdf = pd2pygdf(df)
    #pdf = pygdf.DataFrame().from_pandas(df) # doesn't work
    pca = cumlPCA(n_components=2)
    pca.fit(pdf)
    res = pca.transform(pdf)
    return pdf,res,pca

def array_equal(a,b,threshold=1e-4,with_sign=True):
    a = to_nparray(a)
    b = to_nparray(b)
    if with_sign == False:
        a,b = np.abs(a),np.abs(b)
    res = np.max(np.abs(a-b))<threshold
    return res

def to_nparray(x):
    if isinstance(x,np.ndarray):
        return x
    elif isinstance(x,np.float64):
        return np.array([x])
    elif isinstance(x,pd.DataFrame):
        return x.values
    elif isinstance(x,pygdf.DataFrame):
        return x.to_pandas().values
    elif isinstance(x,pygdf.Series):
        return x.to_pandas().values
    return x

if __name__ == '__main__':
    df = load_mortgage(quarters = 16, rows_per_quarter = 100000)
    #df,res,pca = test_pygdf() 
