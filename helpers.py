def split_in_k(row,k, seed=1):
    chunk_size = len(row[0])//k
    print("chunk size : {}".format(chunk_size))
    np.random.seed(seed)
    indices = np.random.permutation(len(row[0]))
    
    for i in range(k):
        yield [row[0][indices[i * chunk_size: (i + 1) * chunk_size]], row[1][indices[i * chunk_size: (i + 1) * chunk_size]]] 
