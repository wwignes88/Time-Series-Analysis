import sys
import numpy as np
from funcs_.ACF_methods import *
import funcs_.ACF_methods as ACF
from sympy import * 
from sympy.abc import x, y, z
from funcs_.Recursive_predictors import *
import pandas as pd

# set example : 331 = example 3.3.1, 
#               332 = example 3.3.2,
#               534 = example 5.3.4
example = 534 
        


#------------------------
# example 3.3.1 , c.f. section 3.3. 
# use FIRST METHOD to find autocovariance.
if example == 331:
    print('\nExample 3.3.1:\n')
    
    # Find ψj coefficients in expression ψ(z)φ(z) = θ(z)
    # cefficients φ0, φ1, φ2, ..., φ0p in expressions
    #      φ(x) = φ0 - φ1x - φ2x**2 - ... - φpx**p

    φ = np.matrix('1 1 -0.25') 
    θ = np.matrix('1 1')

    
    # find general solution using FIRST method (c.f section 3.6)
    γ = first_method(φ, θ)
    # γ is a function with input t.



#---------------------------------------------
# example 3.3.2; use SECOND METHOD to find autocovariance.
if example == 332:
    print('\nExample 3.3.2:\n')
    
    φ = np.matrix('1 1 -0.25') 
    θ = np.matrix('1 1')
    
    
    # find general solution using SECOND METHOD (c.f section 3.6)
    γ  = second_method(φ, θ)
    # γ is a function with input t.
    
   # note the coefficients in α; 
   #    8, 10.66666667  = 8, 32/3
   # as found in the text.

#---------------------------------------------
if example == 534:

    
    # example 5.3.4
    print('\nExample 5.3.4:\n')
    σ = 1
    φ = np.matrix('1 1 -0.24') 
    θ = np.matrix('1 0.4 0.2 0.1')

    # find general solution using SECOND method (c.f section 3.6)
    γ  = first_method(φ, θ)
    #sys.exit(0)


    # construct K table, pg. 171 (bottom)
    kargs  = γ,σ,φ,θ ; i = 1 ; N = 6
    ktable = True
    if ktable == True: 
        np.set_printoptions(suppress=True)
        Mk = np.zeros([N,N])
        while i <= N:
            j = i
            while j <= N:
                kij = Kij(j,i,kargs) ;
                Mk[j-1,i-1] = np.round(kij,3)
               
                j  += 1
            i += 1
    print(f'\nK (see eq. 5.3.14 in text): \n{Mk}')
    
    # Comparing the results to the table on pg. 171 we see
    # the first method produces results almost identicle to
    # those on pg. 171. The third method is a close second. 
    # the second method is not too far off.
    
    
    

    #------------------------------------
    # now use innovations algorithm to find x^{n+1}
    
    x = np.matrix('1.704 0.527 1.041 0.942 0.555\
                  -1.002 -0.585 0.010 -0.638 0.525')
                  
    table = innovations(x, θ, φ, σ, kargs, 3)
    
    # construct pandas data table

    arrays = np.array(["","","","","","","","","","","",])
    index=["n", "X_{n+1}", "r_n", "θn1", "θn2", "θn3", "X^_{n+1}"]
    table_532 = pd.DataFrame(table,index = arrays, columns=index)
    print(f'\nTable 5.3.2: \n{table_532}')







