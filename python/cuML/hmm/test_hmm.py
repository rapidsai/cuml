import cugraph
import cudf
import time
from scipy.io import mmread
import networkx as nx
import pytest

print('Networkx version : {} '.format(nx.__version__))


def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_Call(M, max_iter, tol, alpha):

    # Device data
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)
    # values = cudf.Series(np.ones(len(sources), dtype = np.float64))
    
    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    t1 = time.time()
    df = cugraph.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Sort Pagerank values
    sorted_pr = []
    for i, rank in enumerate(df['pagerank']):
        sorted_pr.append((i, rank))

    return sorted(sorted_pr, key=lambda x: x[1], reverse=True)


def networkx_Call(M, max_iter, tol, alpha):

    nnz_per_row = {r: 0 for r in range(M.get_shape()[0])}
    for nnz in range(M.getnnz()):
        nnz_per_row[M.row[nnz]] = 1 + nnz_per_row[M.row[nnz]]
    for nnz in range(M.getnnz()):
        M.data[nnz] = 1.0/float(nnz_per_row[M.row[nnz]])

    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # should be autosorted, but check just to make sure
    if not M.has_sorted_indices:
        print('sort_indices ... ')
        M.sort_indices()

    # in NVGRAPH tests we read as CSR and feed as CSC,
    # so here we do this explicitly
    print('Format conversion ... ')

    # Directed NetworkX graph
    Gnx = nx.DiGraph(M)

    z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

    # Networkx Pagerank Call
    print('Solving... ')
    t1 = time.time()

    # same parameters as in NVGRAPH
    pr = nx.pagerank(Gnx, alpha=alpha, nstart=z, max_iter=max_iter*2,
                     tol=tol*0.01)
    t2 = time.time() - t1

    print('Time : ' + str(t2))

    # return Sorted Pagerank values
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)


datasets = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']

Max_Iterations = [500]
tolerance = [1.0e-06]
alpha = [0.85]


@pytest.mark.parametrize('graph_file', datasets)
@pytest.mark.parametrize('max_iter', Max_Iterations)
@pytest.mark.parametrize('tol', tolerance)
@pytest.mark.parametrize('alpha', alpha)
def test_pagerank(graph_file, max_iter, tol, alpha):

    M = ReadMtxFile(graph_file)
    sorted_pr = cugraph_Call(M, max_iter, tol, alpha)
    items = networkx_Call(M, max_iter, tol, alpha)
    # Calculating mismatch
    err = 0
    # assert len(sorted_pr) == len(items)
    for i in range(len(sorted_pr)):
        if(abs(sorted_pr[i][1]-items[i][1]) > tol*1.1):
            err = err + 1
    print(err)
    assert err < (0.01*len(sorted_pr))
