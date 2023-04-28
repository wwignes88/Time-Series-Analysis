import numpy as np
from funcs_.Complex_Gauss import row_reduced
from funcs_.Complex_solve import solve as Isolve
from funcs_.helpful_scripts import find_first_el, Test_zero_matrix
from numpy.linalg import matrix_power

Tol_ = 0.0001
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------

# find generalized null-space of matrix M 

def null_basis(M):  
    DIM = M.shape ; L = DIM[0] ; W = DIM[1] 
    v   = np.matrix(np.zeros(L)).transpose()
    
    # --------------- [Gaussian] row reduction:
    M,E = row_reduced(M)
    
    #--------------------
    x_solve = -np.ones(W)
    # ^^ initialize index array of x vals to be solved for.
    # -1 will let us no not to solve for a given variable (it is dependent so we CHOOSE its value)
    #print(f'[NULL] M: \n{np.round(M)}')
    # find index of first non-zero element in row i:
    i = 0
    rank_ = 0 # also find rank of matrix.
    while i < np.min([L,W]): 
        non_zero   = find_first_el(M[i,:], Tol_) 
        x_solve[i] = non_zero  
        if non_zero >= 0 :
            rank_ += 1 
        i+=1

    # count/ find indices of dependent variables.
    x_dep = [] 
    k = 0 
    while k < len(x_solve):
        if x_solve[k] < 0:
            x_dep.append(k)
        k += 1

    # if no depdendent variables, there is no null space.
    if len(x_dep) == 0:
        return np.matrix(0*np.eye(W)), 0
    
    
    #print(f'\nx_solve : {x_solve}')   
    #print(f'x_dep : {x_dep}')
    #print(f'rank: {rank_}')
    
    k = 0
    while k < len(x_dep):
        x = np.zeros(W).astype('complex128')
        # set one dependent variable at a time to 1. Others remain 0.
        x[x_dep[k]] = 1 + 0j
  
        #print(f'x: {x}')
        
        i = np.min([L,W])-1
        while i >= 0:
            rowi = np.array(M[i,:])[0]
            i_solve = int(x_solve[i]); 
            
            
            if i_solve == W-1: # cannot call x[W:]!!!
                x[i_solve] = v[i]/rowi[i_solve] 
                
            if i_solve >=0 and i_solve < W-1:
                x[i_solve] = (v[i]-np.sum(x[i_solve+1:]*rowi[i_solve+1:]))/rowi[i_solve]
            
            i += -1
            
        # add solution to null space basis set.
        if k == 0: 
            Null_basis = np.matrix(x).transpose()
        if k > 0:
            x_ = np.matrix(x).transpose()
            Null_basis = np.hstack([Null_basis,x_])
        k += 1 
                      # rank of null-space = W-rank_
    return Null_basis, W-rank_
    
    

#===========================================
#======  Find range          basis =========
#===========================================

# pick columns of matrix A that span the range (column space)
# for reduced row form of basis switch 
#           col = A[:,indx] to 
# to 
#           col = M[:,indx]

def range_basis(A):
    A = np.round(A,6) # may cause problems with row reduction if not rounded.
    DIM = A.shape; L = DIM[0] ; W = DIM[1]
    M   = np.matrix(np.copy(A))
    #print(F'\n[range]  M: \n{np.round(M,2)}')
    zeros  = np.zeros(L).astype('complex128')
    zeros  = np.matrix(zeros).transpose()
    solve_ = Isolve(M,zeros, 0)
    x_solve= solve_[3]
    rank   = solve_[2]
    M      = solve_[4] # reduced matrix of A.
    #print(F'M: \n{np.round(M,3)}')
    #print(F'x_solve: {x_solve}')
    #print(F'[range] rank   : {rank}')
    

    if Test_zero_matrix(A) :
        return np.matrix(np.zeros(L)).transpose(), 0

    i = 0
    while i < len(x_solve):
        indx = x_solve[i]
        if indx >= 0:
            col = A[:,indx] # switch A to M if row reduced -- possibly unit -- vectors are desired.
            # (this is not suitable for jordan decomposition though)
            try:
                s = np.hstack([s,col])
            except:
                s = col
        i   += 1
    #input(F's{i}: \n{s}')
    return s, rank


#----------------------------------------------------------------
# input: square matrix. Check if columns span dimension of matrix.
def check_span(B):
    j  = 0 ; N = B.shape[1] ; L = B.shape[0]
    I  = np.matrix(np.eye(L)).astype('complex128') 
    while j < N:
       ej    = I[:,j]
       solve = Isolve(B,ej,1)
       if solve[0]!='dependent':
           raise ValueError(f'Basis does not span e{j}\nx = {solve[:2]}')
       j += 1

#----------------------------------------------------------------   
# check if unit vectors(s) ej are spanned by columns of matrix M.
# If not, append ej's to complete basis. Returns square matrix.
def boil_basis(B):
    N  = B.shape[0]
    I  = np.matrix(np.eye(N))#.astype('complex128') 
    j  = 0 ; span = False
    while B.shape[1] < N:
        ej = I[:,j]
        solve = Isolve(B,ej,1)
        if solve[0]!='dependent':
            B = np.hstack([B,ej])
        x = solve[:1]
        #print(f'M: \n{M}')
        #print(f'x: {x}')
        #print(f'B shape: {B.shape}')
        j += 1 
    if B.shape[0] != N: # should return square matrix
        raise ValueError(f'Basis not boiled to N={N}. B shape: {B.shape[1]}')
    check_span(B)
    return B








