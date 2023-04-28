import numpy as np


#==================================================
#============ TIME-SERIES FUNCTIONS ===============
#==================================================



# presumes t = index, i.e. t = [1,2,3,...] is not
# to be used. Instead use  t = [0,1,2,...]
def lag1_diff(y,t):
    T    = y.shape[0] - 1
    yt   = y[1:t+1,0]
    yt_1 = y[ :t  ,0]
    return yt - yt_1

def regress(X,y):
    # Perform linear regression; β = (X'X)^{-1}Xy:
    XX    = np.matmul(X.transpose(),X)
    XXinv = inverse_(XX)
    Xy    = np.matmul(X.transpose(),y)
    β     = np.matmul(XXinv,Xy)  ; 
    return β,XXinv

#-------------------------
# calculate F statistic
# Rβ - r is matrix equation of restrictions on β. c.f. ch 8 of Hamilton
def F_stat(β,R,r,XXinv,s2):
    m = R.shape[0]
    
    # construct [s**2 R (XX)^(-1)R']^(-1)
    xr  = np.matmul(XXinv,R.transpose())
    rxr = np.matmul(R,xr)
    s2RXRinv  = inverse_(rxr)/s2
    
    # construct Rβ - r:
    Rβ_r = np.matmul(R,β) - r
    
    # F variable:
    F = np.matmul(s2RXRinv,Rβ_r)
    F = np.matmul(Rβ_r.transpose(), F)/m

    return F[0,0]/m


#==================================================
#======= JORDAN DECOMPOSITION &      ==============
#======= ROW REDUCE/ SOLVE FUNCTIONS ==============
#==================================================

from sympy import * 
from sympy.abc import x, y, z

 
# swap rows of matrix.
def row_swap(i1,i2,N): 
    R = np.eye(N)
    R[[i1,i2],:] = R[[i2,i1],:]
    return R

# find first non-zero column in matrix
def find_nonzero_col(M):
    j = 0 ; W = M.shape[1]
    TF = False
    while TF == False:
        colj = M[:,j]
        if np.any(colj != 0):         
            return colj
        j+=1
    raise ValueError('No non-zero columns found.')
    	

# find first non-zero element in a vector
def find_first_el(v,Tol_):
    if type(v)==np.matrix:
        dim = v.shape; w = dim[1]; l = dim[0]
        if w > l:
            v = np.array(v)[0]
    i = 0
    while i<len(v):
        if abs(v[i]) > Tol_:
            return i
        i += 1
    return -1



# shuffle zero rows to top 
def arrange_reduced_mat(A):
    
    DIM = A.shape; L = DIM[0] ; W = DIM[1] 
    count_dep = 0; 
    J_first = np.zeros(L); R = np.eye(L)
    i = 0
    while i<L:
        first_val_indx = np.where(A[i,:]!=0)[0]
        if len(first_val_indx)>0:
            val_i = first_val_indx[0]
        else:
            val_i = L+count_dep
            count_dep += 1
        J_first[i] = val_i
        i +=1
    new_order = np.sort(J_first)
    i = 0
    while i < len(new_order):
        i_swap = np.where(J_first==new_order[i])[0][0]
        if i != i_swap and J_first[i] != new_order[i]:
            Ri = row_swap(i_swap,i,L)
            R  = np.matmul(Ri,R)
            J_first = np.matmul(Ri,J_first)
        i += 1
    M = np.matmul(R,A)
    return M,R
    

#----------- Evaluate scypy str
# evaluate scypy string using lambdify functio.

def evaluate_str(str_,A_,I_):
    Tol_ = 5
    #print(f'[null_transform] I_= \n{np.round(I_)}')
    #print(f'[null_transform] A_= \n{np.round(A_)}')
    #input(f'[null_transform] T_str= {type(A_)}')
    u_mat   = lambdify([x,y],str_)(A_,I_)
    #print(f'[evaluate_str] u_mat= \n{u_mat}')
    u_mat   = np.matrix(np.round(u_mat,Tol_))
    return u_mat




#--------------- Constructing polynomial strings

# construct expanded polynomial str from coeeficient inputs.
# xn is np.array with coefficients ordered as p(A) = x0 + x1*A + x2*X**2 + ...
# where A is the matrix or variable argument.
def pn_string(xn):
    p_expr = f'{xn[0]}*x**{0}'
    j = 1
    while j < len(xn):
        if xn[j] != 0:
            p_expr += f' + {xn[j]}*x**{j}'
        j+=1
    return p_expr


# Find roots of polynomial.
# x is same input as above (see pn_string above) due to np.roots
# function which accepts the order of 

def find_poly_roots(x,rnd):
    x = x[::-1] # reverse order due to np.roots takes first entry as highest power.
    roots_ = np.roots(x)
    Roots = []; Degeneracies = [] ; no_check_list = []
    N = len(roots_)
    # consolodate list (count/ eliminate repeated roots)
    i = 0; 
    while i < N:
        if i not in no_check_list:
            ki = 1
            rooti = np.round(roots_[i],rnd); 
            Roots.append(roots_[i])
            no_check_list.append(i)
            j = i + 1
            while j < N:
                rootj = np.round(roots_[j],rnd)
                if rootj == rooti:
                    ki += 1
                    no_check_list.append(j)
                j += 1
            Degeneracies.append(int(ki))
        i += 1
    return np.array(Roots), np.array(Degeneracies)



#----------- Find inverse of matrix
def inverse_(M):
    import numpy as np
    try:
        return np.linalg.inv(M)
    except:
        raise ValueError("Singular Matrix, Inverse not possible.")

# Perform change of basis on matrix A.
def basis_transform(S,A):
    Sinv = inverse_(S)
    AS   = np.matmul(A,S)
    SAS  = np.matmul(Sinv,AS)
    return SAS
def Reverse_basis_transform(S,A):
    Sinv = inverse_(S)
    AS   = np.matmul(A,Sinv)
    SAS  = np.matmul(S,AS)
    return SAS

#-----------------
# test that T*v == [0]
def Test_null(T,v):
    test = np.matmul(T,v)
    if np.any(abs(test) > 1*1e-5) :
        print(f'test Null: \n{np.round(test,4)}')
        raise ValueError('Tu != 0')
        

#------------------------
# test if entire matrix IS zero       
def Test_zero_matrix(U):
    U = abs(U)
    j = 0; count = 0 ; W =  U.shape[1]
    while j <W:
        if np.all(abs(U[:,j]) < 1*1e-8) :
            count += 1
        j += 1
    if count == W:
        return True
    if count < W:
        return False

# test the equivalenct between two matrices
# if difference is ~ 0 they are taken to be equivalent
# within a tolerance level of TOL
def Test_matrix_equivalence(A,B,TOL):
    U = abs(A-B)
    j = 0; count = 0 ; W =  U.shape[1]
    while j <W:
        if np.all(U[:,j] < TOL) :
            count += 1
        j += 1
    if count == W:
        return True
    if count < W:
        return False

























