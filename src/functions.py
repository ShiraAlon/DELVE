'''FUNCTIONS'''

import pandas as pd
import numpy as np
import random
import scipy
from scipy.spatial.distance import pdist, squareform,euclidean
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse.linalg import eigsh, svds, inv
from scipy.linalg import eig, svd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags, eye
from scipy.spatial import distance_matrix
from sklearn.preprocessing import normalize

# Shnitzer et al
# def diffusion_map(X, c=None):
#     # Pairwise Euclidean distances
#     pairwise_dists = squareform(pdist(X, metric='euclidean'))
#     # Kernel scale ε = median of distances (common practice)
#     epsilon = np.median(pairwise_dists)
#     # Construct affinity matrix using squared distance
#     if c:
#         K = np.exp(- (pairwise_dists ** 2) / ((c*epsilon)** 2))
#     else:
#         K = np.exp(- (pairwise_dists ** 2) / (epsilon ** 2))
#     D_inv = np.diag(1.0 / np.sum(K, axis=1))
#     P = D_inv @ K
#     Q = K @ D_inv
#     # eigvals, eigvecs = eig(P)
#     # idx = np.argsort(eigvals)[::-1]
#     #eigvecs[:, idx], eigvals[idx]
#     return P, Q, K

def diffusion_map(X, adaptive=5500):
    if adaptive is not None:
        K = Kernel_matrix(X, adaptive)
    else:
        # Pairwise Euclidean distances
        pairwise_dists = squareform(pdist(X, metric='euclidean'))
        # Kernel scale ε = median of distances (common practice)
        epsilon = np.median(pairwise_dists)
        # Construct affinity matrix using squared distance
        K = np.exp(- (pairwise_dists ** 2) / (epsilon ** 2))
    D_inv = np.diag(1.0 / np.sum(K, axis=1))
    P = D_inv @ K
    Q = K @ D_inv
    # eigvals, eigvecs = eig(P)
    # idx = np.argsort(eigvals)[::-1]
    #eigvecs[:, idx], eigvals[idx]
    return P, Q, K
    
# compute a kernel matrix with adaptive bound
def Kernel_matrix(df, epsilon):
    n_cells = df.shape[0]
    #euclidian distance matrix:
    D = squareform(pdist(df,'euclidean'))
    dist_sort = np.sort(D,axis = 1)
    sigmas = dist_sort[:, epsilon]#np.mean(dist_sort[:, 1:epsilon+1], axis=1)
    Sig = np.outer(sigmas,sigmas)#sigmas*sigmas
    kernel_matrix = np.exp(-(D**2)/Sig)
    return kernel_matrix

# compute Laplacian graph (type=random walk): L = D^(-1)K
def LG_RW(W, k=None):
    D = np.diag(np.sum(W, axis=1))
    D_inv = np.diag(1 / np.sum(W, axis=1))
    
    L_rw = np.eye(W.shape[0]) - D_inv @ W
    
    d, v = np.linalg.eig(L_rw)
    
    idx = np.argsort(d.real)   # ascending
    d = d[idx].real
    v = v[:, idx].real
    
    if k:
        v = v[:, :k]
        d = d[:k]
    
    return L_rw, d, v

# compute Laplacian graph (type=regular): L = D - K
def LG_K(W, k = None, initial = None):
    
    D = np.diag(np.sum(W, axis=1))    
    Lrw = D - W
    
    d, v = np.linalg.eigh(Lrw)
    idx_ = np.argsort(d)[::-1]
    v = v[:,idx_]
    if k:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v


# compute Laplacian graph (type=Symmetric): L = D^(-0.5)K D^(-0.5)
def LG_sym(W, k = None, initial = None):
    
    D = np.diag(np.sum(W,axis = 1)**(-0.5))
    Lrw = D@W@D
    d, v = np.linalg.eigh(Lrw)
    idx_ = np.argsort(d)[::-1]
    v = v[:,idx_]
    d = d[idx_]
    if k != None:
        v = v[:, :k]
        d = d[:k]
    return Lrw, d, v



def spectral_mapping(L,param,function_type):
    #d,v = np.linalg.eigh(L):
    d = L.shape[0]
    if function_type == 'inv':
        return np.linalg.inv(L+param*np.eye(d))
        
        
def circ_convolution(x,y):
    x_ext = np.concatenate((x,x))
    a = np.correlate(x_ext,y,) #,mode = 'same'
    return a
        
    

def calc_differential_vec(L_A, v_B, k, Q=None):
    
    U1 = v_B[:,:k]
    Q1 = U1 @ U1.T
    Q1 = np.eye(Q1.shape[0]) - Q1
    Q1 = Q1@L_A@Q1

    s, u1 = np.linalg.eigh(Q1)
    idx_order = np.argsort(s)[::-1]
    u1 = u1[:,idx_order]
    if Q:
        return Q1 ,s, u1
    else:
        return s, u1



def calc_distance(x, y):
    n = len(x)
    shift_positions = np.arange(-n + 1, n)
    shift = shift_positions[np.argmax(abs(np.correlate(x,y)))]
    x_roll = np.roll(x, shift)
    dist = euclidean(y, x_roll)
    
    return dist

def calc_sig_to_noise(x, y, s,display=True, sort=True):
    #x = estimate, y = true
    if sort:
        sig = pd.Series(x[np.argsort(y)])
    else:
        sig = pd.Series(x) 
    
    amp = np.mean(np.abs(sig.rolling(window=s).mean())) #- sig.rolling(window=s).mean().mean()
    print(sig.rolling(window=s).mean().mean())
    noise = np.mean(np.power(sig.rolling(window=s).std(),2))
    sig_noise = amp/noise
    
    vecs = np.abs(sig.rolling(window=s).mean() - sig.rolling(window=s).mean().mean())/np.abs(sig.rolling(window=s).std())
    p_val = len(np.where(vecs<=1)[0])/len(vecs)
   
    if display:
        # calculate a 60 day rolling mean and plot
        sig.rolling(window=s).mean().plot(style='k')
        # add the 20 day rolling standard deviation:
        sig.rolling(window=s).std().plot(style='b')
        plt.show()
        print("signal to noise: "+str(sig_noise))
        plt.hist(vecs, bins=100)
        plt.show()
        print("P value of signal to noise: " + str(p_val))
        
    
   
    return sig_noise



        