import sys
import glob
import numpy as np
import pandas as pd
from   funcs_.helpful_scripts import *
import matplotlib.pyplot as plt

print('case 4: \nconstant term and time trend; \n\
      true process is random walk with drift')
      
print(f'\nRegression Model of real GNP:\nyt = α + ρy_(t-1) + δt + u_t')



#==================================================
#============= LOAD DATA ==========================
#==================================================

path  = "data_files"
files = glob.glob(path + "/*.csv")


df = pd.read_csv('data_files\\realGNP.csv',
        header = 0,
        usecols=["GNP"]) ; #print(f'\n-----\ndf = \n{df}')
s  = df['GNP'].to_numpy()
s  = 100*np.log(s) ; 

v=s


# define arrays
yt_arr = v[1:] ; 
yt_1   = np.matrix(v[:len(v)-1])
T      = len(yt_arr)   ; 
t_arr  = np.arange(1,T+1)


#================================================= eqn. 17.4.56
print('\n-------------------------\n\
eqn 17.4.56:\n')

# construct X matrix in regression 
#    y_t = α + ρy_{t-1} + u_t = βX + u_t
# where ut is an i.i.d variable ~N[0,σ^2]
X      = np.matrix(np.ones(T))
X      = np.vstack([X,yt_1]) ; 
X      = np.vstack([X,t_arr]) 
X      = X.transpose()

# convert yt, yt_1 to column matricies
yt_1  = yt_1.transpose()
yt    = np.matrix(yt_arr).transpose()

# Perform linear regression; β = (X'X)^{-1}Xy:
XX    = np.matmul(X.transpose(),X)
XXinv = inverse_(XX)
Xy    = np.matmul(X.transpose(),yt)
β     = np.matmul(XXinv,Xy)  ; 
print(f'\nβ = \n{β}')
α,ρ,δ = np.array(β.transpose())[0][:]

# calculate error:
ut  = yt - α - ρ*yt_1 - δ*np.matrix(t_arr).transpose()
ut2 = np.multiply(ut,ut)

# residual:
s2  = np.sum(ut2)/(T-2) ; s = np.sqrt(s2)
print(f'\ns**2 = {s2}')

# standard deviation:
σ2  = s2*XXinv[1,1] ; σ = np.sqrt(σ2) # 1,1 is index of ρ coefficient in X 
print(f'σ    = {σ}')


# t test: 
print(f'\n------------\nt test (ρ=1)')
t_OLS  = (ρ-1)/np.sqrt(s2*XXinv[1,1]) ; print(f't    = {t_OLS}')
tc     = 3.44 # critical value
if abs(t_OLS) > tc:
    print(f'   reject Ho: ρ=1')

if abs(t_OLS) < tc:
    print(f'   accept Ho: ρ=1')



# F test (δ=0, ρ=1)
print('\n\n------------\nF test: δ=0, ρ=1')

R = np.matrix('0 0 1;\
               0 1 0')
r = np.matrix('0;\
               1')

F   = F_stat(β,R,r,XXinv,s2)  ; 
print(f'F = {np.round(F,4)}')
Fc  = 6.42 # critical value
if abs(F) >= Fc:
    print(f'   reject Ho: ρ=1')

if abs(F) < Fc:
    print(f'   accept Ho: ρ=1')





#------------------------------------------------------- PLOT
# plot data/ regression
plt.plot(t_arr,yt,'b',t_arr,α + ρ*yt_1 + δ*np.matrix(t_arr).transpose(), 'r')
plt.title("blue: actual data\
          red  : regression")






#============================================
#============================================
#============================================
print(f'\n-----------------------------------\n\
example 17.7:\n\
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
print(f'\nPhillips-Perron statistic:\n{pp}')

# OLS t-statistic w/ serial correlation, c.f. eqn 17.6.12
t_OLS_serial = np.sqrt(γ0/λ2)*t_OLS  -  0.5*((λ2-γ0)/λ)*(T*σ/s)
print(f'\nt OLS (serially correlated):\n{t_OLS_serial}')

# t test:
tc = 2.89
if abs(t_OLS_serial) > tc:
    print(f'   reject Ho: ρ=1')

if abs(t_OLS_serial) < tc:
    print(f'   accept Ho: ρ=1')










