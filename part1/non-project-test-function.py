import time
import numpy as np
import scipy.sparse as sparse

ITER = 100
K = 10
N = 10000

def naive(indices, k):
    mat = [[1 if i == j else 0 for j in range(k)] for i in indices]
    M = np.array(mat).T
    #print(M)
    return M


def with_sparse(indices, k):
    n = len(indices)
    M = sparse.coo_matrix(([1]*n, (indices, range(n))), shape=(k,n)).toarray()

    #print(M)
    return M

def with_sparsenp(indices, k):
    n = len(indices)
    M = sparse.coo_matrix((np.ones(n), (indices, range(n))), shape=(k,n)).toarray()

    #print(M)
    return M


n, d, k = 5, 5, 7
Y = np.random.randint(0, K, size=N)
t0 = time.time()
for i in range(ITER):
    M=naive(Y, K)

#print(M)
print(time.time() - t0)


t0 = time.time()
for i in range(ITER):
    M=with_sparse(Y, K)
#print(M)
print(time.time() - t0)

t0 = time.time()
for i in range(ITER):
    M=with_sparsenp(Y, K)
#print(M)
print(time.time() - t0)
