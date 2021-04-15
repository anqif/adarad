import numpy as np

def perc_nz(A):
        A_tot = A.shape[0] * A.shape[1]
        A_perc_nz = A.nnz / float(A_tot)
        return A_perc_nz

def select_cols(A, ncol, method = 'uniform'):
        A_cols = A.shape[1]
        if ncol > A_cols:
                raise ValueError("Must have ncol <= {}".format(A_cols))
        if method is 'first':
                return A[:,:ncol]
        if method is 'uniform':
                step = int(float(A_cols)/ncol)
                A_tmp = A[:,::step]
                return A_tmp[:,:ncol]   # Necessary if A_cols not a multiple of ncol
        elif method is 'random':
                indices = np.arange(A_cols)
                np.random.shuffle(indices)
                return A[:,indices[:ncol]]
        else:
                raise ValueError("method must be 'first', 'uniform', or 'random'")

def col_block_rows(A, blocks, method = "mean"):
	m, n = A.shape
	m_new = len(blocks)
	b_sum = np.sum(blocks)
	if b_sum > m:
		raise ValueError("Sum over blocks must be <= {0}".format(m))

	ptr = 0
	A_col = np.zeros((m_new, n))

	if method is "mean":
		for i in range(m_new):
			A_col[i,:] = A[ptr:(ptr + blocks[i]),:].mean(axis = 0)
			ptr = ptr + blocks[i]
	elif method is "sum":
		for i in range(m_new):
			A_col[i,:] = A[ptr:(ptr + blocks[i]),:].sum(axis = 0)
			ptr = ptr + blocks[i]
	else:
		raise ValueError("method must be 'mean' or 'sum'")
	return A_col

