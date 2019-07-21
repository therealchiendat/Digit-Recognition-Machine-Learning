import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    K=(np.dot(X,Y.T)+c)**p
    return K



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    '''
    A=np.linalg.norm(X) **2
    B=np.matmul(X,Y.T)
    C=np.linalg.norm(Y) **2
    print(A)
    print()
    print(B)
    print()
    print(C)
    inner=A-2*B+C
    K=np.exp(-gamma*(inner))
    '''
    inner=(X[np.newaxis,:, :] - Y[:, np.newaxis, :])
    K = np.exp(-gamma * np.sum(inner**2,axis=2)).T
    return K
