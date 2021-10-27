import numpy as np
import scipy as sp
# n_x_n tri_diag decomp

# func: LU finder
# param: a length n-1(lower diag), d length n(center diag), c length n-1(upper diag)
# return: two matrices in order L, then U
def find_LU(a,d,c):
    E = np.insert([a[i]/d[i] for i in range(len(a))],0,1)
    d_prime = np.concatenate(([d[0]],[d[i+1]-c[i]*(a[i]/d[i]) for i in range(len(c))]))
    L = np.vstack((np.concatenate(([1],np.zeros(len(E)-1))),[np.insert(np.insert(np.zeros(len(E)-2),i,E[i]),i+1,1) for i in range(len(E)-1)]))      
    U = np.vstack(([np.insert(np.insert(np.zeros(len(c)-1),i,d_prime[i]),i+1 , c[i]) for i in range(len(c))],np.concatenate((np.zeros(len(c)),[d_prime[-1]]))))
    return L, U

#Ex. for matrix
#  
#         136  90    0    0    
#   A =   90   98    -67  0
#         0    -67   132  46
#         0    0     46   17
#

find_LU([90,-67,46],[136,98,132,17],[90,-67,46])