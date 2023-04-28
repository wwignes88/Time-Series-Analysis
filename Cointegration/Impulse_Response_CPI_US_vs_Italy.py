import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   funcs_.helpful_scripts import *
from funcs_.Jordan_Decompose import Jordan_form
from funcs_.null_range_basis import *
np.set_printoptions(suppress=True)
from numpy.linalg import matrix_power

print('Impulse response for  U.S. vs Italy Consumer price index\n')
# c.f. pg 586 and pg. 19
#==================================================
#============= LOAD DATA ==========================
#==================================================

path  = "data_files"
files = glob.glob(path + "/*.csv")

#------------------
# first data set:
df1 = pd.read_csv('data_files\\CPIA_UCSL_1973.csv',
        header = 0,
        usecols=["CPIAUCSL"]) ; #print(f'\n-----\ndf = \n{df}')
s1  = df1['CPIAUCSL'].to_numpy()
v1  = 100*np.log(s1) 
v1  = v1 - v1[0]      
yt1 = np.matrix(v1).transpose()


#-----------------
# second data set:
df2 = pd.read_csv('data_files\\CPI_ITALY_1973.csv',
        header = 0,
        usecols=["ITACPIALLMINMEI"]) ; #print(f'\n-----\ndf = \n{df}')
s2  = df2['ITACPIALLMINMEI'].to_numpy()
v2  = 100*np.log(s2) ; 
v2  = v2 - v2[0]
yt2 = np.matrix(v2).transpose()

#----------------
# third data set:
df2 = pd.read_csv('data_files\\US_Italy_exchange_rate_1973.csv',
        header = 0,
        usecols=["EXITUS"]) # print(f'\n-----\ndf = \n{df}')
s3  = df2['EXITUS'].to_numpy()
v3  = 100*np.log(s3) ; 
v3  = v3 - v3[0]
yt3 = -np.matrix(v3).transpose()
# take negative value because data file contaions Lira to dollar
# conversion rates, but we want dollar to Lira



#=======================================================
#================ SET PARAMETERS =======================
#=======================================================

# set regression array y; 
# y = ζΔy_{t-1} + ... + ζΔy_{t-p} + α + ρ*yt_1 + δ*ct
# so here we choose the vector y = = pt y_t (c.f. eqn.19.2.1 )
y = yt1              # for eqn. 19.2.1
y = yt1 - yt2 - yt3  # for eqn. 19.2.2

# include deterministic time trend δ*ct
# (optional, *compare eqn.19.2.1 and eqn.19.2.2)
d_trend = True  # for eqn. 19.2.1
d_trend = False # for eqn. 19.2.2


# define time array 
T      = len(yt1)
t_arr  = np.matrix(np.arange(1,T+1)).transpose()


#---------------------
# set parameter values.
Δ = 1    # lag value
t = np.max(t_arr) - 1
p = 12       # so we'll go to Δy_{t-12} = y_12 - y_13
X_indx = p+1 # index at which X matrix will begin



#=======================================================
#========= CONSTRUCT LAG [REGRESSION] MATRIX  ==========
#=======================================================

# initialize arrays
yt_1   = y[X_indx-1:t,0]   # y_{t-1}
ct     = t_arr[X_indx:,0]  # truncated (or 'cut') time array (or column matrix)
t_indx = ct.shape[0]-1     # index of maximal t value in truncated time array
cT     = ct.shape[0]       # total number of observations used in regression.

# to stack the appropriate length vector Δy{t-j}  = y_t - t_{t-1}
# onto regression matrix X, we use Xindx-j-1 where -1 is 
# to account for python indexing.
X = lag1_diff(y, t-1)[X_indx-1-1:,0] # initialize X matrix
j        = 2
while j <= p:
    Δyj    = lag1_diff(y, t-j)[X_indx-j-1:,0]
    X      = np.hstack([X, Δyj])
    j     += 1

# stack ones for α [constant] term.
ones   = np.matrix(np.ones(ct.shape[0])).transpose()
X      = np.hstack([X,ones])
α_indx = X.shape[1] - 1 # index of α in β array (see below)


# stack ρy_{t-1} column 
X = np.hstack([X,yt_1])
ρ_indx = X.shape[1] - 1 # index of ρ in β array (see below)

# stack t column for deterministic δt term (optional)
if d_trend == True:
    X      = np.hstack([X,ct])
    δ_indx = X.shape[1] - 1 # index of δ in β array (see below)

yt = y[X_indx:,0] # truncate yt1



#---------------------------------------------
# Perform linear regression; β = (X'X)^{-1}Xy:
β,XXinv = regress(X,yt)
#print(f'β [9]regression coefficients] : \n{β}')

α  = β[α_indx ,0]
ρ  = β[ρ_indx ,0]



#============================================
#========= Impulse Response Coeff. ==========
#============================================

print('\n\n-------------------------')
p = 13
test = False
if test :
    β = np.matrix('0.55 , -0.06,  0.07,  0.06,\
                   -0.08, -0.05, -0.17, -0.07,\
                   0.24 , -0.11,  0.12,  0.05').transpose()

# construct φj coefficients 
# this is my own way of solving φ_j using the determined
# ζ_j coefficients. Pretty straight forward; equations 19.1.37
# and 19.1.38 form a triangular system that is easily solved
# by starting at the last coefficient; φ_p = -ζ_{p-1}.
j  =  p-1 # start @ j=p-1 for python indexing.
φj = -β[j-1,0] ; φ=np.matrix(φj) ;  j+=-1
while j > 0:
    φj   = -β[j-1,0]-np.sum(φ)
    φ    =  np.append(φ, [[φj]], axis = 0)
    j   += -1
# The last coefficient to solve for is φ1 (or φ0 in python indexing)
# we solve this from 19.1.7
φ1 = ρ - np.sum(φ)  
φ  = np.append(φ,[[φ1]], axis = 0)

# We have constructed the φ array in the order [φ_p,φ_{p-1},...,φ2,φ1]'
# where ' denotes it is a single column matrix.
# so now we reverse the order and transpose into a single row matrix
# in preparation for constructing the matrix F (see pg. equation 1.2.3) 
φ = φ[::-1].transpose()

I = np.matrix(np.eye(p))[:p-1,:].astype('complex128') 
F = np.vstack([φ,I])
print(f'\nF = \n{np.round(F,5)}')
#sys.exit(0)
J,M, λ, K, X = Jordan_form(F) # Jordan form of F
print(f'\nJ = \n{np.round(J,5)}')





#sys.exit(0)
# now to find the impulse response coefficients ψj
# c.f. pgs. 9-10 and 586
j_arr = np.array([]) ; ψ = np.array([]) 
j = 1
while j < 64:
    Jj = matrix_power(J,j) # jth power: J**j
    Fj = Reverse_basis_transform(M,Jj)
    ψj = Fj[0,0]
    ψ  = np.append(ψ,ψj)
    j_arr  = np.append(j_arr,j)
    j  += 1

plt.figure() 
plt.plot(j_arr,ψ)
plt.title("ψj [Impulse response]")























