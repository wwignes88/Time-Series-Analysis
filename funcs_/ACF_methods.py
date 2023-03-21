import numpy as np
from funcs_.Complex_solve import solve as Isolve
from   sympy import * 
from   sympy.abc import x, y, z

#=========================================================
#============== FINDING ARMA COEFFICIENTS ψ ==============
#=========================================================

# c.f. section 3.3. Find ψj coefficients in expression
#          ψ(z)φ(z) = θ(z)
# which is an inversion of the ARMA model.
# see pg 91

# inputs: 
#    φ = np.matrix('φ0, φ1, φ2,...,φp') ; φ0 = 1
#    φ = np.matrix('θ0, θ1,...,θq')     ; θ0 = 1

def solve_ψ_ARMA(θ,φ):
    q = θ.shape[1] - 1; 
    p = φ.shape[1] - 1; print(f'p,q = {p,q}')
    ψ = np.matrix([1])
    globals()[f'ψ0'] = 1
    j = 1
    
    # construct ψj = θj + Σφψ_{j-k} ; 0<k<=j
    #                                 0<=j< max(p,q+1)
    while j < np.max([p, q+1]):
        if j > q:
            θj = 0
        if j <= q:
            θj = θ[0,j]
        ψj = θj 
        k  = 1
        while k <= j:
            if k > p:
                φk = 0
            if k <= p:
                φk = φ[0,k]
            ψj_k  = ψ[0,j-k]
            ψj   += ψj_k*φk
            k    += 1
        ψ = np.append(ψ,[[ψj]], axis = 1)
        globals()[f'ψ{j}'] = ψj
        j += 1
    return ψ


#=========================================================
#============== FINDING AUTOCOVARIANCE γ(K) ==============
#=========================================================



#============== FIRST METHOD (see pg. 91) ================

# find general solution of homogoneous linear difference 
# equation with constant coefficients using the FIRST method
# presented on pg. 91

def first_method(φ, θ):
    p = φ.shape[1] - 1  ;  q = θ.shape[1] - 1
    
    ψ = solve_ψ_ARMA(θ,φ).transpose()
    print(f'ψ      = \n{ψ}')
    indx = np.max([p,q]) - p
    print(f'p = {p}, q = {q}, ind = {indx}')
    
    #-----------------------------------------------------
    # find roots of φ(x)
    rnd = 3 # round root calculation 
    # *Note: here φ coefficients need adjustment. For the expression,
    #        φ(x) = φ0 - φ1x - φ2x**2 = 0
    
    # we now now need to account for the negative signs on φ1,φ2,...φp.
    x = np.copy(φ); x[0,1:] = - x[0,1:]
    
    # np.roots (used by 'find_poly_roots') accepts polynomial 
    # coefficients in descending order, i.e.
    # highest order coefficient first.
    x = x[0,::-1]     ; print(f'x reversed = {x}')
    # all of my functions accept matrix inputs.
    x = np.matrix(x)  ; print(f'\nx          = {x}')
    
    φRoots, φDegeneracies  = find_poly_roots(x,3)
    print(f'\nφRoots       : {φRoots}')
    print(f'φDegeneracies: {φDegeneracies}')
    # *Note: 'degeneracies' is a physics term. It should be 'multiplicities'.

    #------------------------------------------------------
    k = len(φRoots) # number of distinct ξ (roots)
    
    t0,tmax  = np.max([p,q+1]) - p , np.max([p,q+1])   
    ψ0 = ψ[t0:,0] # initial conditions.
    
    # construct matrix .... initial conditions
    t = t0
    while t < tmax:
        rowt = []
        i = 0
        while i < k:
            ξi = φRoots[i] ; ri = φDegeneracies[i]
            j  = 0 
            while j < ri:
                rowt.append((t**j)*(ξi**(-t)))
                j += 1
            i += 1
        if t == t0:
            M = np.matrix(rowt)
        if t > t0:
            M = np.vstack([M,np.matrix(rowt)])
        t += 1

    print(f'\ninitial conditions:\nψ0     = \n{ψ0}')
    print(f'M  = \n{M}')
    
    print('\nsolving Mα = ψ0....')
    α = Isolve(M,ψ0,5)[1] ; α = np.real(α)
    print(f'**α [coefficients] = {α}\n')

    #------------------------------------------
    # construct scypy string for γ(k)
    i = 0 ; indx_ = 0 ; ψn_str = f'('
    while i < k:
        ξi = np.round(φRoots[i],3) ; ri = φDegeneracies[i]
        j  = 0 
        while j < ri:
            # z = t. scypy does not recognize 't'
            αj  = np.round(α[0,indx_],3)
            ψn_str  += f'{αj}*z**{j}'
            indx_ += 1
            if j != ri -1:
                ψn_str  += '+'
            j += 1
        ψn_str += f')*{ξi}**(-z)'
        if i != k-1:
            ψn_str += '+('
        i += 1
    print(f'ψn_str = {ψn_str}\n')
    
    # find autocovariance (c.f. equation 3.3.1)

    def γ(k):
        return first_meth_autocovariance(k,ψn_str,p,q)
    k  = 0 ; p  = φ.shape[1]-1 ;  q = θ.shape[1]-1
    while k <= p:
        γk = γ(abs(k))
        print(f'γ{k} = {np.round(γk,4)}')
        k += 1
    return γ

#--------------------------------------------
# for the FIRST METHOD we have,
#   γ(k) = σ**2 Σψj*ψ{j+|k|} (top of pg. 91)  
# its an infinite sum, but it should converge for finite q, 
# so we simply set a tolerance for detecting the final value.

def first_meth_autocovariance(k,ψn_str,p,q):
    
    ψn = lambdify(z,ψn_str) # z = t. scypy does not recognize 't'

    γk = 0 ; γlast = 0
    j  = 0 ; tol = 0.000000001; kill = False
    while j < 100:#kill == False:
        if j < np.max([p,q+1]):
            ψj = globals()[f'ψ{j}']
        if j >= np.max([p,q+1]):
            ψj = ψn(j)   
        
        if j+abs(k) < np.max([p,q+1]):
            ψjk = globals()[f'ψ{j+abs(k)}']
        if j+abs(k) >= np.max([p,q+1]):
            ψjk = ψn(j+abs(k))   
            
        γj  = ψj*ψjk
        γk += γj
        j  += 1
        if abs(γlast - γj) < tol and j > 0:
            return γk
        else:
            γlast = γj
        if j == 1000:
            raise ValueError('[first-way] γk is not converging')
            return np.nan





#============== SECOND METHOD (see pg. 92) ================

# find general solution of homogoneous linear difference 
# equation with constant coefficients using the SECOND method.

def second_method(φ, θ):
    p = φ.shape[1] - 1
    q = θ.shape[1] - 1
    
    ψ = solve_ψ_ARMA(θ,φ)
    print(f'ψ   = {ψ}')
  
    
    #-----------------------------------------------------
    # find roots of φ(x)
    rnd = 3 # round root calculation 
    # *Note: here φ coefficients need adjustment. For the expression,
    #        φ(x) = φ0 - φ1x - φ2x**2 = 0
    
    # we now now need to account for the negative signs on φ1,φ2,...φp.
    x = np.copy(φ); x[0,1:] = - x[0,1:]
    
    # np.roots (used by 'find_poly_roots') accepts polynomial 
    # coefficients in descending order, i.e.
    # highest order coefficient first.
    x = x[0,::-1]     ; print(f'\nx reversed = {x}')
    # all of my functions accept matrix inputs.
    x = np.matrix(x) 
    φRoots, φDegeneracies  = find_poly_roots(x,5)

    print(f'\nφRoots       : {φRoots}')
    print(f'φDegeneracies: {φDegeneracies}')

    # *Note: 'degeneracies' is a physics term. It should be 'multiplicities'.

    #-------------------------------------------------------
    # construct matrix M to solve Mα = φ(B)γ(k) = Σ θj ψ{j-k}
    # where the sum runs over j ; 0<j<=j ; 0<=k<=max{p,q+1}
    # see bottom of pg. 92. Note that what I'm calling α is in the 
    # text referred to as β -  the coefficient in the general solution.
    
    k = len(φRoots)
    initial_cond = np.matrix([])
   
    t0,tmax  = np.max([p,q+1]) - p , np.max([p,q+1])    
    t0,tmax = 0,p #!!! these initial time values are wrong!?!?!?!
    
    t  = t0
    while t < tmax:
        rowt = []
        i = 0
        while i < k:
            ξi = φRoots[i] ; ri = φDegeneracies[i]
            j  = 0 ;
            while j < ri:
                sum_ = (t**j)*(ξi**(-t))
                l = 1
                while l  <= p:
                    tl    = abs(t-l)
                    sum_ += -φ[0,l]*(tl**j)*(ξi**(-tl))
                    l    += 1
                rowt.append(sum_) 
                j += 1
            i += 1

        if t == t0:
            M = np.matrix(rowt)
        if t > t0:
            M = np.vstack([M,np.matrix(rowt)])
            
        #------------- --------------
        # construct initial conditions. c.f. ew. 3.3.8
        initial_sum = 0
        j = t
        while j <= q:
            initial_sum += θ[0,j]*ψ[0,j-t]
            j += 1
        initial_cond = np.append(initial_cond,[[initial_sum]], axis = 1)
        t += 1

    print(f'\ninitial conditions:  {np.round(initial_cond,3)}')
    print(f'M  = \n{np.round(M,4)}')
    
    print('\nsolving Mα = v...')
    α = Isolve(M,initial_cond.transpose(),0)[1] ; α = np.real(α)
    print(f'α = {α}')
    

    #------------------------------------------
    # construct scypy string for γ(k)
    i = 0 ; indx_ = 0 ; γt_str = f'('
    while i < k:
        ξi = np.round(φRoots[i],4)
        ri = φDegeneracies[i]
        j  = 0 
        while j < ri:
            # z = t. scypy does not recognize 't'
            αj  = np.round(α[0,indx_],4)
            γt_str  += f'{np.round(αj,3)}*z**{j}'
            indx_   += 1
            if j != ri -1:
                γt_str  += '+'
            j += 1
        γt_str += f')*{np.round(ξi,3)}**(-z)'
        if i != k-1:
            γt_str += '+('
        i += 1
    
    # --------------------------------------------
    # functionalize γ(t), print out first p values.
    print(f'\nγ(t)  = {γt_str}\n')
        
    def γ(t):
        γk  = lambdify(z,γt_str)
        return γk(abs(t)) 
    
    k  = 0 ;
    while k <= tmax:
        γk = γ(k)
        print(f'γ{k} = {np.round(γk,4)}')
        k += 1
    return γ


#============== THIRD METHOD (see pg. 97) ================


def third_method(φ,θ):
    γdict = {}
    p = φ.shape[1] - 1
    q = θ.shape[1] - 1
    
    # ARMA coefficients:
    ψ = solve_ψ_ARMA(θ,φ)
    
    
    M = np.matrix(np.zeros([p+1,p+1]))
    t = 0  ; initial_cond = []
    while t <= p:
        
        #------------------------- 
        # initial condition
        Σψt = 0 
        j = t
        while j <= q:
            Σψt += θ[0,j]*ψ[0,j-t]
            j += 1
        initial_cond.append(Σψt)

        
        #--------------------------
        # construct matrix
        M[t,t] = 1
        j = 1
        while j <= p:
            M[t,abs(t-j)] += -φ[0,j]
            j += 1
        t += 1
    print(f'M : \n{M}')
    
    
    print('\nsolving Mγ = ψ0....')
    initial_cond = np.matrix(initial_cond).transpose()
    γ = Isolve(M,initial_cond,0)[1] ; γ = np.real(γ)
    #print(f'γ = {np.round(γ,4)}\n')
    
    i = 0
    while i < γ.shape[1]:
        globals()[f'γ{i}'] = γ[0,i]
        γdict[f'γ{i}'] = np.round(γ[0,i],4)
        i += 1
    
    def γ(t):
        t = abs(t)
        
        if t <= p:
            return globals()[f'γ{t}']
        if t > p:
            k = p + 1
            while k <= t:
                l  = 1 ; γk = 0
                while l <= p:
                    γk_l = globals()[f'γ{k-l}']
                    γk  += φ[0,l]*γk_l
                    l   += 1
                globals()[f'γ{k}']  = γk
                k += 1
            return γk

    k  = 0 ;
    while k <= p:
        γk = γ(k)
        print(f'γ{k} = {np.round(γk,4)}')
        k += 1
    return γ







#=========================================================
#===== FINDING POLYNOMIAL ROOTS/ MULTIPLICITIES ==========
#=========================================================

#--------------------------------------------
# Finding polynomial roots

# find where item occurs in a list
def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

# solve expanded polynomial w/ numpy roots function. Then locate/ count/ delete
# repeated values. returns list of roots and their multiplicities (or  in physics
# terminology, the degeneracies). Among mathematicians what I am calling the 
# degeneracy would usually be referred to as the algebraic multiplicity of the 
# eigenvalue....least this is what I was thinking when I coded it. As it turns out,
# it is the geometric multiplicity which is synonomous with degeneracy.

# Note that scypy will not factor an expression if it has to resort to floats.
# Ergo, constructing the polynomial from the roots is the most tobust way to 
# go about constructing a scypy string to evaluate.

# input: x is single row matrix of polynomial coefficients, e.g. 
#        p(x) = α2x**2 + α1x + α0
# is entered as x = np.array([α2,α1,α0])

def find_poly_roots(x,rnd):
    x = np.array(x)[0] # convert matrix to array
    roots_ = np.round(np.roots(x),rnd)
    i = 0; Roots  = [] ; Degeneracies = []
    no_check_list = []
    
    # consolodate list (count/ eliminate repeated roots)
    while i < len(roots_):
        rooti  = roots_[i]; 
        
        if rooti not in no_check_list:
        
            # find repeated roots:
            repeats = find_indices(roots_, rooti)
            Roots.append(rooti)
            Degeneracies.append(len(repeats))
            
            # update no-check list so we don't count this root again
            no_check_list.append(rooti)
        
        i += 1
    return Roots, Degeneracies



