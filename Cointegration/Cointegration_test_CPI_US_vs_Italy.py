
import time
import glob
import numpy as np
import pandas as pd
from   funcs_.helpful_scripts import *
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# Test null hypothesis of Cointegration for 
# U.S. vs Italy Consumer price index
print('Cointegration test for U.S. vs Italy Consumer price index\n')
print(f'\nRegression model:\n #   y = ζΔy_(t-1)+ ... + ζΔy_(t-p) + α + ρ*y_(t-1) + δ*ct + ut')

# see example on pgs. 583-585

#==================================================
#============= LOAD DATA ==========================
#==================================================

path = "data_files"
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
df3 = pd.read_csv('data_files\\US_Italy_exchange_rate_1973.csv',
        header = 0,
        usecols=["EXITUS"]) # print(f'\n-----\ndf = \n{df}')
s3  = df3['EXITUS'].to_numpy()
v3  = 100*np.log(s3) ; 
v3  = v3 - v3[0]
yt3 = -np.matrix(v3).transpose()
# take negative value because data file contaions Lira to dollar
# conversion rates, but we want dollar to Lira


#=======================================================

# set regression array y; 
# y = ζΔy_{t-1} + ... + ζΔy_{t-p} + α + ρ*yt_1 + δ*ct
# so here we choose the vector y = = pt y_t (c.f. eqn.19.2.1 )
y = yt1              # for eqn. 19.2.1
y = yt1 - yt2 - yt3  # for eqn. 19.2.2

# include deterministic time trend δ*ct
# (optional, *compare eqn.19.2.1 and eqn.19.2.2)
d_trend = True  # for eqn. 19.2.1
#d_trend = False # for eqn. 19.2.2


# define time array 
T      = len(yt1)
t_arr  = np.matrix(np.arange(1,T+1)).transpose()


#---------------------
# set parameter values.
Δ = 1    # lag value
t = np.max(t_arr) - 1
p = 12       # so we'll go to Δy_{t-12} = y_12 - y_13
X_indx = p+1 # index at which X matrix will begin

#-----------------------------------------------------
# initialize arrays/ construct lag regression matrix 
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

#=======================================

# *debugging print statements [ignore me]
Print = False 
if Print == True:
    print(f' ct      : \n{ct[:3, 0]}\n\
      ... \n{ct[t_indx-3:, 0]}')
    print(f'\n X: \n{X[0:3,:]}\n\
      ... \n{X[t_indx-3:,:]}')
      
    print(f' yt1     : \n{yt1[:5, 0]}\n\
  ... \n{yt1[t_indx:, 0]}')
    print(f'\n yt     : \n{yt[:3, 0]}\n\
  ... \n{yt[t_indx-2:, 0]}')
    print(f'\n yt_1     : \n{yt_1[:3, 0]}\n\
  ... \n{yt_1[t_indx-2:, 0]}')

#=================================================
# calculate regression
# Perform linear regression; β = (X'X)^{-1}Xy:
β,XXinv = regress(X,yt)
print(f'\nregression coefficients : \n{β}')


#==================================================
# calculate OLS t statistic (c.f. equation 17.4.29)
print('\n------------\nOLS t statisic:')

# construct estimated regression
# y = ζΔy_{t-1} + ... + ζΔy_{t-p} + α + ρ*yt_1 + δ*ct
j = 1; y_est = 0
while j   <= p:
    Δyj    = lag1_diff(y, t-j)[X_indx-j-1:,0]
    y_est += β[j-1,0]*Δyj
    j     += 1
α  = β[α_indx ,0]
ρ  = β[ρ_indx ,0]
y_est += α + ρ*yt_1 
if d_trend == True:
    δ      =  β[δ_indx,0]
    y_est +=  δ*ct

# calculate s**2 (c.f. 17.4.30)
nβ  = β.shape[0]
ut  = yt - y_est          ; ut2 = np.multiply(ut,ut)
s2  = np.sum(ut2)/(cT-nβ) ; s   = np.sqrt(s2)
#print(f'\ns**2  = {np.round(s2,5)}')

# calculate σ
σ2 = s2*XXinv[ρ_indx, ρ_indx]
σ  = np.sqrt(σ2)
print(f'σ     = {np.round(σ,5)}')

# OLS t-statistic
t_OLS = (ρ-1)/σ
print(f't_OLS = {np.round(t_OLS,5)}')

# critical value for comparison:
tc = 3.44     # for equation 19.2.1
if d_trend == True:
    tc = 2.88 # for equation 19.2.2
    
# peform Dickey-Fuller t-test:
# testing for cointegration is equivalent to testing
# if zt is I[0], i.e. zt is I[1] (a unit root exists)
# means the series are NOT cointegrated.
# see discussion on pg. 582
if abs(t_OLS) <= tc:
    print('\nAccept  Ho: unit root (ρ=1)\n\
          p, p* are NOT cointegrated')
if abs(t_OLS) > tc:
    print('\nReject Ho: unit root (ρ!=1)\n\
          p, p* ARE cointegrated')




#============================================
#================= Plotting =================
#============================================

# plot CPI data
plt.figure() 
plt.plot(t_arr,yt1,t_arr,yt2,'r',t_arr,yt3,'k--' )
plt.title("Consumer Price Index (CPI)\n\
          blue: U.S.\
          red : Italy\
          --  : exchange rate")

# plot zt = pt - st = pt*
plt.figure() 
plt.plot(ct,yt,'r',ct,y_est,'b--')
plt.title("zt = pt - st - pt*\n\
          blue--:regression\
          red   : zt")




print('\n*Initializing Phillips-Perron test (serial correlation)...')
time.sleep(3)
#============================================
#========= Phillips-Perron tests ============
#============================================
# c.f. pg 585
print(f'\n\n=============================\
      \n\nPhillips-Perron test (serial correlation):' )

# perform regression yt = α + ρ*yt_1 + ut
X = ones                # stack ones column for constant α 
X = np.hstack([X,yt_1]) # stack ρy_{t-1} column 
ρ_indx = X.shape[1] - 1 # index of ρ in β array (see below)

β,XXinv = regress(X,yt) ; print(f'\nβ = \n{β}')
α,ρ = np.array(β.transpose())[0][:]

# find error
ut  = yt - α - ρ*yt_1
ut2 = np.multiply(ut,ut)
s2  = np.sum(ut2)/(cT-2) ; s = np.sqrt(s2)
print(f'\ns**2  = ({s})**2')

# calculate σ
σ2  = s2*XXinv[1,1] ; σ = np.sqrt(σ2)
print(f'σ     = {σ}')

# OLS t-statistic
t_OLS = (ρ-1)/σ
print(f't_OLS = {np.round(t_OLS,5)}')


# define autocovariance, c.f. eqn. 17.6.16:
γ0 = np.sum(np.multiply(ut,ut))/T
j  = 0 ;  q = p ; γarr = np.array([])
while j <= q:
    γ = 0
    t = j+1
    while t < cT:
        γ += ut[t,0]*ut[t-j,0]
        t += 1
    γ = γ/cT
    γarr = np.append(γarr, γ)
    j += 1
        
print(f'\nγ vals = {np.round(γarr,4)}')  

# Newey-West estimator
λ2 = γ0
j = 1
while j <= q:
    λ2 += 2*(1-j/(q+1))*γarr[j]
    j  += 1
λ = np.sqrt(λ2)
print(f'\nNewey estimator: λ**2 = {λ2}')

# Estimated variance, c.f eqn 17.4.30


#----------------------------------------------
# Phillips-Perron t-tests:
    
# Phillips-Perron Zp statistic, c.f. eqn 17.6.8
print(f'\n-----Zp test:')
Zp = T*(ρ-1) - 0.5*((T**2)*σ2/s2)*(λ2 - γ0)
print(f'Zp = {np.round(Zp,4)}')
tc = 13.9
if abs(Zp) < tc:
    print('Accept  Ho: unit root (ρ=1) exists\n\
            ==> p, p* are NOT cointegrated')
if abs(Zp) >= tc:
    print('Reject Ho: unit root (ρ!=1) exists\n\
            ==> p, p* ARE cointegrated')



# Phillips-Perron Zp statistic, c.f. eqn 17.6.8
# OLS t-statistic w/ serial correlation, c.f. eqn 17.6.12
print(f'\n-----Zt test:')
Zt = np.sqrt(γ0/λ2)*t_OLS  -  0.5*((λ2-γ0)/λ)*(T*σ/s)
print(f'Zt = {np.round(Zt,4)}')
tc = 2.88
if abs(Zt) < tc:
    print('Accept  Ho: unit root (ρ=1) exists\n\
            ==> p, p* are NOT cointegrated')
if abs(Zt) >= tc:
    print('Reject Ho: unit root (ρ!=1) exists\n\
            ==> p, p* ARE cointegrated')






























