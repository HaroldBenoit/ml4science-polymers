import numpy as np
def split_in_k(y,row,k, seed=1):
    chunk_size = len(row)//k
    np.random.seed(seed)
    indices = np.random.permutation(len(row))
    
    for i in range(k):
        index=indices[i * chunk_size: (i + 1) * chunk_size]


        yield (y,row[index])
