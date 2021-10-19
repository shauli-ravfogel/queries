import numpy as np
import faiss

num_vecs=1000
dims = 512
M = 32  # The number of sub-vector. Typically this is 8, 16, 32, etc.
nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
# Param of IVF
nlist = int(np.sqrt(num_vecs))  # The number of cells (space partition). Typical value is sqrt(N)
# Param of HNSW
hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32
# Setup
quantizer = faiss.IndexHNSWFlat(dims, hnsw_m)
index = faiss.IndexIVFPQ(quantizer, dims, nlist, M, nbits)
print("here")
index.train(np.random.randn(10000, dims).astype("float32"))

