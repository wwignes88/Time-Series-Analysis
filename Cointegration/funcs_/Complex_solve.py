import numpy as np
from funcs_.Complex_Gauss import row_reduced
Tol_ = 0.0001
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------


# M is matrix. v is row TRANSPOSED single row matrix (so a COLUMN vector).
# we solve the equation Mx = v. x_dep_val is to set the
# value of dependent variables. The code can easily be 
# manipulated to allow user input for these variables.
# if M is known to be full rank, then just enter any value
# for x_dep_val - it won't be used.
# returns: x - a ROW vector in matrix form. 
# Note the discrepency between the forms of input vector v and output
# vector x. This preference is arbitrary and can be changed according
def solve(M,v,x_dep_val):
    DIM = M.shape ; L = DIM[0] ; W = DIM[1] ; 
        
    
    if L != len(np.array(v)):
        raise ValueError('[Complex_solve.py] length of matrix input does not equal vector solution length')
    	
    # --------------- [Gaussian] row reduction:

    M,E,v = row_reduced(M,v)

    #--------------------- Check if solution exists/ count dependent variable
    
    # check if solution exists: if a row of the reduced matrix is all zeros yet
    # the corresponding element of v is non-zero, this results in division by zero.
    i = 0
    while i < L:   
        if np.all(abs(M[i,:]) == 0) and abs(v[i]) != 0: # No solution
            rank_ = 0
            x = np.zeros(W)*np.nan
            x_solve = -1*np.ones(W)
            return 'NA', x, rank_ , x_solve, M, E
        i += 1  
   
        
    # ----------------------
# set to zero. Edit values as needed later.
    x_solve = -np.ones(W)
    # ^^ initialize index array of x vals to be solved for

    # find x_solve: the indices of x variables to be solved for @ each row of M.
    i = 0
    rank_ = 0 # also find rank of matrix.
    while i < np.min([L,W]): 
        # find index of first non-zero element in row i:
        non_zero = find_first_el(M[i,:], Tol_) # returns -1 if no [absolute] non-zero val detected
        x_solve[i] = non_zero  
        if non_zero >= 0 :
            rank_ += 1 
        i+=1

    # Initialize solution
    x = np.zeros(W).astype('complex128')*np.nan 
    # set dependent variables to x_dep_val input. 
    # Alternatively, allow user input.
    i = 0
    dep = False # for detecting if dependent variables exist
    while i < W:
        if i not in x_solve:
            dep = True
            x[i] = x_dep_val #Alternative: input(f'set dep. var. val: x[{i}] = ')
        i += 1
    
    #print(F'\n[Complex_solve.py]  M: \n{np.round(M,2)}')
    #print(F'[Complex_solve.py] x_solve: {x_solve}')
    #print(F'[Complex_solve.py] A: {A}')
    #input(F'[Complex_solve.py] [range] rank   : {rank}')
    
    
    # start @ bottom for upper triangular matrix.             
    i = np.min([L,W])-1
    while i >= 0:
        rowi = np.array(M[i,:])[0]
        i_solve = int(x_solve[i]) # index of ind. variable to be solved

        if i_solve == W-1: # cannot call x[W:]!
            x[i_solve] = v[i]/rowi[i_solve] 
            
        if i_solve >=0 and i_solve < W-1:
            x[i_solve] = (v[i]-np.sum(x[i_solve+1:]*rowi[i_solve+1:]))/rowi[i_solve]
        
        i += -1

    if np.all(abs(x)==0) == True and dep == False:
        return 'independent', x , rank_ , x_solve.astype('int'), M, E

    if np.any(np.isnan(x)==True):
        return 'NA', x, rank_ , x_solve.astype('int'), M, E

    else:
        return 'dependent', x, rank_ ,  x_solve.astype('int'), M, E

    
#===================================
#===================================



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








