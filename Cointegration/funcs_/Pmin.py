# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:27:26 2022

@author: Wayne
"""
import sys
from funcs_.Complex_Gauss import row_reduced
from funcs_.Complex_solve import solve as Isolve
from   sympy import * 
from   sympy.abc import x, y, z
import numpy as np
from   funcs_.helpful_scripts import *

Sleep_toggle = 0
input_toggle = 0


#-----------------------------------------------------------------
# find minimum polynomial of a [square] matrix.
# inputs: A [square np.matrix], rnd : rounding parameter [integer]
def find_Pmin(A):
    A_ = np.matrix(A)
    DIM = A.shape ; L = DIM[0] ; W = DIM[1] 

    # define unit vectors en:
    i = 0
    while i < W:
        ei = np.zeros(W)
        ei[i] = 1
        ei = np.matrix(ei).transpose()
        globals()['e'+str(i+1)] = ei
        i += 1

    # initialize parameters
    I_     = np.matrix(np.eye(W)).astype('complex128')   # [complex] Identity matrix
    zeros_ = np.matrix(np.zeros(W)).transpose()          # for solving Pmin*x = [0]
    Pn     = np.matrix(np.copy(I_)).astype('complex128') # start w/ Pmin = I (identity matrix)
    X = [] # polynomial factor(s) coefficient list, to be updated for each en vector eliminated.
    n = 1 
    ROOTS = np.array([]); DEGENERACIES = np.array([])
    while  n <= W: 
        #print(f'\n----------')

        en  = globals()['e'+str(n)]   
        ui  = np.matmul(Pn,en)  
        Un  = ui ; 
        
        Dep = False ; i = 1
        while Dep == False:
            ui = np.matmul(A_,ui)
            Un = np.hstack([Un,ui])   

            # Test if non-zero x_n exists such that Un*x_n = 0 [linear dependence]
            Test = Isolve(Un, zeros_,1) # dependent variables set to 1
            
            # we seek the FIRST power of matrix A such that A**i * ui = 0
            if Test[0] == 'dependent':                
                #print(f'\n----------- n={n} \n')
                en = globals()[f'e{n}']
                
                # coefficients of [expanded] Pn factor
                xn = Test[1]
                #print(f'x{n} = {xn}')
                X.append(xn) # append list of polynomial factors

                Pexpanded_str = pn_string(xn)
                #print(f'Pexpanded_str : {Pexpanded_str}')
                pn = evaluate_str(Pexpanded_str,A_,I_)
                Pn = np.matmul(Pn,pn)   
                
                test = np.matmul(Pn,en)
                if np.all(test != 0):
                    raise ValueError('Pmin{n} does not eliminate the e{n} unit vector')
                	

                Roots, Degeneracies = find_poly_roots(xn,4)
                ROOTS        = np.append(ROOTS,Roots)
                DEGENERACIES = np.append(DEGENERACIES,Degeneracies)

                
                #print(f'\np{n} roots (λ vals)  : {np.round(Roots,3)}')
                #print(f'p{n} Multiplicies    : {Degeneracies}')

                Dep = True # exit loop
            i += 1 
        #print(f'\nPmin ROOTS         : {np.round(ROOTS,3)}')
        #print(f'Pmin Multiplicities: {DEGENERACIES}')



        #-------------------------------
        # Find next n value using the criteria Pn(A)*en ≠ 0


        j = n+1; Elim = True       
        while Elim   == True and j <= W:
            ej        = globals()[f'e{j}']

            # check if  Pmin(A)*ej ≠ 0
            check   = np.matmul(Pn,ej)
            if np.all(abs(check) == 0):
                u9 = 5
                #print(f'Pmin(A)*e{j} == 0')

            else: 
                #print(f'Pmin(A)*e{j} ≠ 0')
                Elim = False
                
                # Fail-safe: hopefully we'll never reach n=W without finding 
                # Pmin such that Pmin(A)*eW ≠ 0 
                if n == W:
                    raise ValueError("[Jordan_decomposition.py] No Pmin found that yeilds {0} matrix!!!!. check math")
        
                break
            j  += 1 ; 
        n = j; #print(f'\nsetting n={n}...');
        #sleep(1)


    #----------------------------
    #sleep(2)
    #print('\n====================')
    # Print final polynomial, pn coefficients list, and check Pmin(A) == 0.

    Pmin = Pn
    if Test_zero_matrix(Pmin) == False:
        raise ValueError(f'Pmin is non-zero: Pmin = \n{Pmin}')
    #print('\nPmin(A) (should be {0} matrix) ' + f':\n{Pmin}')
    #print(f'\nPmin    : {Pmin_str}')
    #print(f'\npolynomial factor coefficients:\nX: {X}')

    # Retrun the reverse order of pn coeffients list so X = [xn,x+{n-1},...,x1]
    # corresponding to PN = Pmin = pn*...*p1. Being as univariate matrix polynomials commute
    # it actually does not matter. Nevertheless, it represents the order in which Pmin was 
    # actually constructed.

    return X , ROOTS, DEGENERACIES














