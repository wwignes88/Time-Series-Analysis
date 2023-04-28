import sys
import glob
import numpy as np
import pandas as pd
from funcs_.helpful_scripts import *
import matplotlib.pyplot as plt


# example 17.4 and 7.6 of Hamilton, pg.  494

print('case 2: \nconstant term but no time trend; \n\
      true process is random walk')
print(f'\nRegression Model of U.S. Treasure Bills:\nyt = α + ρy_(t-1) + u_t')
#!!!!! Note: data file has missing entries, so we had to 
# customize the loading of data for this one.

#==================================================
#============= LOAD DATA ==========================
#==================================================

path  = "data_files"
files = glob.glob(path + "/*.csv")


df = pd.read_csv('data_files\\TreasuryBills.csv',
        header = 0,
        usecols=["RATE"]) ; #print(f'\n-----\ndf = \n{df}')
s  = df['RATE'].to_numpy()


# the csv file for Treasury Bills has some unreadable inputs
# so we have to find a way to work around them.
count_errors = 0 ; v = np.array([]) 
i = 0
while i < len(s):
    try:
        v = np.append(v,float(s[i]))
    except:
        count_errors += 1
    i += 30 # adjust day count to 90 for quarterly data.
#print(f'count_errors = {count_errors}')
#print(f'data count   = {len(v)}')

# define time array
t  = np.arange(1,len(v)) ; t_arr = np.copy(t)



#================================
print('\n-------------------------\n\
example 17.4: \n\
t-test on Ho: ρ=1 for U.S. Treasure Bill data ')



# this is the regression prior to a stationary transformation (see READ_ME file)

# construct arrays for X
yt_arr = v[1:] ; T = len(yt_arr)
yt_1   = np.matrix(v[:len(v)-1])
ones   = np.ones(len(yt_arr))

# construct X matrix in regression 
#    y_t = α + ρy_{t-1} + u_t = βX + u_t
X      = np.matrix(ones)
X      = np.vstack([X,yt_1]) ; yt_1 = yt_1.transpose()
X      = X.transpose()
yt     = np.matrix(yt_arr).transpose()

# Perform linear regression; β = (X'X)^{-1}Xy:
β,XXinv = regress(X,yt); 
print(f'\nβ [regression coefficients] = \n{β}')
α,ρ     = np.array(β.transpose())[0][:]

# find error
ut  = yt - α - ρ*yt_1
ut2 = np.multiply(ut,ut)

# squared residual:
s2  = np.sum(ut2)/(T-2) ; s = np.sqrt(s2)
print(f'\ns**2 = {s2}')

# standard deviation:
σ2  = s2*XXinv[1,1] ; σ = np.sqrt(σ2)
print(f'σ    = {σ}')

# OLS t-statistic:
# see eqn. 17.4.29
# accept H0: ρ=1 (unit root) if t<tc. Fc is the critical  
# value in table B.6
t_OLS = ((ρ-1))/σ 
print(f'\nOLS statistic:\nt    = {np.round(t_OLS,4)}')


# t test:
tc = 2.89
if abs(t_OLS) > tc:
    print(f'\nreject Ho: ρ=1')

if abs(t_OLS) < tc:
    print(f'\naccept Ho: ρ=1')

          
#===============================================================


#===============================================================
# plot data/ regression
t  = np.arange(1,len(v)) ; t_arr = np.copy(t)
plt.plot(t,yt,'b',t,α + ρ*yt_1, 'r')
plt.title("U.S. real GNP (c.f. figure 17.2, pg. 503 of Hamilton\n\
          blue: actual data\
          red  : regression")
          
          
          
          


#============================================
#============================================
#============================================
print(f'\n-----------------------------------\n\
example 17.6:\n\
Phillips-Perron test for unit root using residual regression to handle serial correlation:')
# define autocovariance, c.f. eqn. 17.6.16:
γ0 = np.sum(np.multiply(ut,ut))/T
j  = 0 ;  q = 4 ; γarr = np.array([])
while j <= q:
    γ = 0
    t = j+1
    while t < T:
        γ += ut[t,0]*ut[t-j,0]
        t += 1
    γ = γ/T
    γarr = np.append(γarr, γ)
    j += 1
        
print(f'\nγ0,γ1,γ2,... = {γarr}')  

# Newey-West estimator
λ2 = γ0
j = 1
while j <= q:
    λ2 += 2*(1-j/(q+1))*γarr[j]
    j  += 1
λ = np.sqrt(λ2)
print(f'\nNewey estimator: λ**2 = {λ2}')

# Estimated variance, c.f eqn 17.4.30

# Phillips-Perron p statistic, c.f. eqn 17.6.8
pp = T*(ρ-1) - 0.5*((T**2)*σ2/s2)*(λ2 - γ0)
print(f'\nPhillips-Perron statistic:\nρ = {pp}')

# OLS t-statistic w/ serial correlation, c.f. eqn 17.6.12
t_OLS_serial = np.sqrt(γ0/λ2)*t_OLS  -  0.5*((λ2-γ0)/λ)*(T*σ/s)
print(f'\nt OLS (serially correlated):\nt = {t_OLS_serial}')


# t test:
tc = 2.89
if abs(t_OLS_serial) > tc:
    print(f'   nreject Ho: ρ=1')

if abs(t_OLS_serial) < tc:
    print(f'   accept Ho: ρ=1')













