import argparse
import faiss
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import subprocess
import tqdm
import pickle
import random

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def add_to_index(index, vecs, cosine = False):

    vecs = np.array(vecs).astype(np.float32)
    #if cosine:
    vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True).astype(np.float32)
    #index.add(np.ascontiguousarray(vecs).astype("float32"))
    if IDS:
      index.add_with_ids(vecs, np.array(range(len(vecs))))
    else:
      index.add(vecs)

def index_vectors(similarity_type, dims=None ,num_vecs=None, vecs=None):

        random.seed(0)
        np.random.seed(0)
        if vecs is None:
            vecs = np.random.randn(num_vecs, dims)
        
        if similarity_type == "cosine" or similarity_type == "dot_product":       
            index = faiss.IndexFlatIP(dims)
        elif similarity_type == "lsh":
            index = LSHIndex(vecs, range(len(vecs)))
            #index = faiss.IndexHNSWFlat(dims, 32) #32 is the number of HNS2 neighbors. no need for training in this index.
        elif similarity_type == "old":
            index = ExactIndex(vecs, range(len(vecs)))
        else:
            raise Exception("Unsupported metric.")
        index.build()
        return index
        

class LSHIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
    def build(self, num_bits=8):
        self.index = faiss.IndexLSH(self.dimension, num_bits)
        self.index.add(self.vectors)
        
    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k) 
        # I expect only query on one vector thus the slice
        return [self.labels[i] for i in indices[0]]
        
class ExactIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
     def build(self):
        self.index = faiss.IndexFlatL2(self.dimension,)
        self.index.add(self.vectors)
        
    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k) 
        # I expect only query on one vector thus the slice
        return [self.labels[i] for i in indices[0]]
                
if __name__ == "__main__":
 
        parser = argparse.ArgumentParser(description='balanced brackets generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--fname', dest='fname', type=str,
                        default="output.mean-cls.jsonl")
        parser.add_argument('--similarity_type', dest='similarity_type', type=str,
                        default="cosine")  
        parser.add_argument('--num_vecs', dest='num_vecs', type=int,
                        default=None)
        parser.add_argument('--dims', dest='dims', type=int,
                        default=None)
                                                                                                     
        args = parser.parse_args()
        random.seed(0)
        np.random.seed(0)
        IDS = args.similarity_type != "old"
        index = index_vectors(args.similarity_type, args.dims, args.num_vecs)
