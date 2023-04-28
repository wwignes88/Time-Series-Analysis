import numpy as np
from funcs_.helpful_scripts import arrange_reduced_mat
Tol_ = 1e-10


def pivot(M,i):
    L = len(M)
    R = np.matrix(np.eye(L)).astype('complex128')
    
    col  = M[i:,i]
    max_ = np.max(abs(np.real(col)))
    try:
        max_ind = np.where(abs(np.real(col))==max_)[0][0]+i
        
    except:
        max_ind = i
    
    if max_ < Tol_:
        
        max_ = np.max(abs(np.imag(col)))
        if max_ < Tol_:
            return False, M, np.eye (L, dtype=complex)
        try:
            max_ind = np.where(abs(np.imag(col)==max_))[0][0]+i
        except:
            max_ind = i

    R[[max_ind,i],:] = R[[i,max_ind],:]
    
    return True, M, R


    
#----------------------------
def eliminate_col(M, Ri, i):
    DIM = M.shape ; L = DIM[0] ; W = DIM[1] ;

    # initiate elementary row operation matrices.
    Ei  = np.eye (L, dtype=complex)

    M = np.matmul(Ri,M)
    
    Ei[i,i] = 1/M[i,i] ; M = np.matmul(Ei,M)
    Ei = np.matmul(Ei,Ri)


    ER = np.eye (L, dtype=complex)
    EI = np.eye (L, dtype=complex)

    j = 0
    while j < L:
        if j != i :
            EI[j,i] = - np.imag(M[j,i])*1j
            ER[j,i] = - np.real(M[j,i])
        j += 1
 
    Ei = np.matmul(EI,Ei); M = np.matmul(EI,M)
    Ei = np.matmul(ER,Ei); M = np.matmul(ER,M)
    
    return M, Ei


#---------------------------
def row_reduced(M_,v_= None):
    if type(M_) != np.matrix :
        raise ValueError("[Complex_Gauss.py] solve func accepts matrix inputs with column vectors")
    
    # convert to float64 (handles division better)
    M  = np.copy(M_).astype('complex128')#.astype(np.float64)
    M  = np.matrix(M)

    v  = np.copy(v_).astype('complex128')#.astype(np.float64)
    v  = np.matrix(v_) # * note: v_ should already be transposed

    # if v input, create augmented matrix M|v
    try:     
        if type(v_) != np.matrix:
            raise ValueError("[Complex_Gauss.py] row_reduced func accepts column vector inputs for v")
        
        a_ = v_[1] # will create error if no v_ input
        Mv = np.hstack([M,v])

        solving = True
    except: 
        solving = False

        Mv = M 
    
    DIM = Mv.shape ; L = DIM[0] ; W = DIM[1]
    
    # initialize elementary row operation matrix.
    E = np.eye(L).astype('complex128')  
    j = 0; E_str = ''
    while j < np.min([W,L]): 

        TF,Mv,Rj = pivot(Mv,j) 
        # *Note: pivot returns False if all entries @ or below diagonal are ~0.
        # we do not want to divide by zero!

        if TF == True: # 

            Mv,Ej =  eliminate_col(Mv, Rj, j)
  
            E = np.matmul(Ej,E)
            E_str += F'E{j}'

            j += 1
        else:
            j += 1   

    if solving == True:
        # recover matrix M and solution v from augmented matrix Mv
        M = Mv[:,:W-1]  ; v = Mv[:,W-1]
        return M,E,v
    else:
        M = Mv[:,:W]  ;
        return M,E
    
    

















