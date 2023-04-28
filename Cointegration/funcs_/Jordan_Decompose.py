# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:27:26 2022

@author: Wayne
"""
import sys
from funcs_.null_range_basis import *
from funcs_.Complex_solve import solve as Isolve
from funcs_.Pmin import find_Pmin
from   sympy import * 
from   sympy.abc import x, y, z
import numpy as np
from   funcs_.helpful_scripts import *
#np.set_printoptions(suppress=True)
from numpy.linalg import matrix_power


# example from 'Time Series Analysis' by James D. Hamilton
# this is the matrix used to find the impulse response function 
# for the example presented on pgs. 585-586. c.f. eqn. 1.2.3.
# As for the relevance of the Jordan Form, see eqn. 1.2.42 
# see 'cointegration\impulse_response' file in github for calculation 
# of matrix itself.
#
A = np.matrix(' 1.318 -0.367  0.029  0.032 -0.04 ;\
  1.     0.     0.     0.     0.   ;\
  0.     1.     0.     0.     0.   ;\
  0.     0.     1.     0.     0.   ;\
  0.     0.     0.     1.     0.   ')

# example from 
# Attila M´at´e. 'The Jordan canonical form'. 
# Brooklyn College of the City University of New York, 2014.
B = np.matrix(' 0 1   0  0 -1 -1 ;\
               -3 8   5  5  2 -2 ;\
               1  0  -1  0 -1  0 ;\
               4 -10 -7 -6 -3  3 ;\
               -1 3   2  2  2 -1 ;\
               -2 6   4  4  2 -1')



# Calculate the Jordan form of matrix A (square matrix)
# see 'Linear Algebra Done Right' by Axler, pgs. 271-273 and 
# procedure 2.31 

def Jordan_form(A):
    DIM = A.shape; L = DIM[0]; W = DIM[1] 
    
    #------------------------
    # define unit vectors en:
    i = 0
    while i < W:
        ei = np.zeros(W)
        ei[i] = 1
        ei = np.matrix(ei).transpose()
        globals()['e'+str(i+1)] = ei
        i += 1


    # ---------------------------------
    # Find minimal polynomial factor coefficients : a list of vectors corresponding
    # to Pmin(A) = p1(A)p1(A)...pn(A)
    X, λ, K = find_Pmin(A)
    #print(f'X = {X}')
    #print(f'λ = {np.round(λ,3)}')
    #print(f'K = {K}')
    
    
    I_ = np.matrix(np.eye(W)).astype('complex128') 
    S  = np.matrix(np.copy(I_))


    #--------------------------------------------------------------------
    # For each n<N value, construct T*u where
    # T = (A-λ[i]), u = Π_i T_i ; i≠n
    # where   Tn = (A-λn}**kn ; kn = multiplicity of λn
    #A = row_reduced(A)[0]
    
    N  = A.shape[1] ; Nλ = len(λ)
    n  = 0
    while n < Nλ:
        #print(f'\n------------------λ{n}')
        #---------------- Construct u
        u = I_ ; i = 0
        while i < Nλ:
            if i  != n:
                Ti = matrix_power(A-λ[i]*I_, int(K[i]))
                u  = np.matmul(u,Ti)
            i += 1
        
        #---------------- Construct T
        T  = A-λ[n]*I_
        Tk = matrix_power(T, int(K[n]))
        TN = matrix_power(T, N)
        Tu = np.matmul(T,u)
        Test_null(Tk,u)  # test T**k *u == [0]
        
        #--------------- Construct Range basis of u
        u_range_basis, u_range_rank = range_basis(u)
        i  = 1
        un = u_range_basis[:,0] ; λ_basis = un ; Tun = un
        while i < K[n]:
            Tun     = np.matmul(T,Tun)
            λ_basis = np.hstack([Tun, λ_basis])
            i += 1
        if n == 0:
            λ_Basis = λ_basis
        if n > 0:
            λ_Basis = np.hstack([λ_Basis,λ_basis])
        n += 1
    
    Nλ = λ_Basis.shape[1]; λ_Basis0 = np.copy(λ_Basis)
    
    # append unit vectors to form a basis, then convert them to null-space vectors
    # (see axler, pg 272 )
    if Nλ < N:
        λ_Basis = boil_basis(λ_Basis) # append unit vectors to form spanning basis
        j = Nλ
        while j < N:
            wj  = λ_Basis[:,j]
            Twj = np.matmul(T,wj)
            xj  = Isolve(T,Twj,1)[1] ; xj = np.matrix(xj).transpose()
            # now set u_{n+j} = wj - xj = ej = xj
            unj = wj - xj   ;  Test_null(T,unj) 
            λ_Basis[:,j] = unj
            j += 1
        check_span(λ_Basis)
        #print(f'λ_Basis: \n{np.round(λ_Basis,2)}')
    
    
    J = basis_transform(λ_Basis,A)
    #print(f'\nJ = \n{np.round(J,5)}') 
    
    # check result:  
    A_ = Reverse_basis_transform(λ_Basis,J)
    #print(f'\nF = M*J*Minv = \n{np.round(F1,5)}') 
    equivalence = Test_matrix_equivalence(A,A_,0.0001)
    if equivalence == False:
        raise ValueError('Jordan Decomposition did not produce approximately equivalent transformation:\n\
                          Input : A = \n{A}\n\
                          Output: S*J*S^(-1) = \n{A_}')


    
    return J,λ_Basis, λ, K, X
    
    
Jordan_form(A)
Jordan_form(B)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    