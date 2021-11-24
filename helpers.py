import numpy as np
def split_in_k(y,row,k, seed=1):
    chunk_size = len(row[0])//k
    print("chunk size : {}".format(chunk_size))
    np.random.seed(seed)
    indices = np.random.permutation(len(row[0]))
    
    for i in range(k):
        index=indices[i * chunk_size: (i + 1) * chunk_size]
        yield y[index],[row[0][index], row[1][index]] 
