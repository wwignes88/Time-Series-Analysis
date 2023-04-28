import sys
import glob
import numpy as np
import pandas as pd
from   funcs_.helpful_scripts import *
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# Test null hypothesis of Cointegration for 
# real quarterly personal disposable income (yt)
# and real personal consumption expenditures (ct)
print('Leads AND lags model: Cointegration test for Personal Disposable Income & Personal Consumption Expenditure\n')
print(f'\nRegression model:\ny = ζ_1Δy_(t-4)+ ... + ζ_5Δy_(t) +...+ζ_9Δy_(t+4)+ α + ρ*y_t + ut')

# c.f.example on pg 610 (also 600)
#==================================================
#============= LOAD DATA ==========================
#==================================================

path  = "data_files"
files = glob.glob(path + "/*.csv")


start_date = '7/1/1948'
end_date   = '10/1/1988'

#------------------
# first data set:
df1 = pd.read_csv('data_files\\real_personal_consumption_expenditures_US.csv',
        header = 0,
        usecols=["DATE","PCECC96"]) ; #print(f'\n-----\ndf = \n{df}')
dates1 = df1['DATE'].to_numpy()
begin_indx = np.where(dates1 == start_date)[0][0]
end_indx   = np.where(dates1 == end_date)[0][0]
s1  = df1['PCECC96'].to_numpy()
v1  = 100*np.log(s1) 
#v1  = v1 - v1[0]      
yt1 = np.matrix(v1).transpose()[begin_indx:end_indx+1,0]


#-----------------------------------------------
# second data set:
df2 = pd.read_csv('data_files\\real_personal_disposable_income_US.csv',
        header = 0,
        usecols=["DPIC96"]) ; #print(f'\n-----\ndf = \n{df}')
s2  = df2['DPIC96'].to_numpy()
v2  = 100*np.log(s2) ; 
#v2  = v2 - v2[0]
yt2 = np.matrix(v2).transpose()[begin_indx:end_indx+1,0]



#================================================
# Perform regression on y = ζΔy_(t-1) + α + ρy_(t-1)

y1 = yt1  # set y1 = yt1 = ct (real personal consumption expenditures )
y2 = yt2  # set y2 = yt2 = yt (real personal disposable income )

# define time array 
T      = len(y1) # number of time values taken.
t      = T-1      # index of maximal time value
t_arr  = np.matrix(np.arange(1,T+1)).transpose()


#------------------------------------------------
# set parameter values.
p = 4        # so we'll go to Δy_(t-p) = y_(t-p) - t_(t-p-1)
t_max = t-p  # maximal time value that can be taken for lead p.


d_trend = True # include time trend δt

#------------------------------------------------
# initialize arrays/ construct lag difference regression matrix 
ct     = t_arr[p+1:t_max+1,0]  # truncated (or 'cut') time array (or column matrix)
cT     = ct.shape[0]
t_indx = ct.shape[0]-1     # index of maximal t value in truncated time array



# to stack the appropriate length vector Δy_(t-j) = y_(t-j) - t_(t-j-1)
# onto regression matrix X, we use Xindx-j-1 where -1 is 
# to account for python indexing.
j  = -p 
X  = lag1_diff(y2, t_max+j)[p+j:,0] # initialize X matrix
j += 1
while j <= p:
    Δyj  = lag1_diff(y2, t_max+j)[p+j:,0]
    X    = np.hstack([X, Δyj])
    j   += 1
ζlag     = X.shape[1] - 1 # index of [maximal] ζ Δy difference coefficient in β array (see below)


# stack ones for α [constant] term.
ones   = np.matrix(np.ones(ct.shape[0])).transpose()
X      = np.hstack([X,ones])
α_indx = X.shape[1] - 1 # index of α in β array (see below)


# stack ρy_t column 
yt = y2[p+1:t_max+1,0]
X  = np.hstack([X,yt])
ρ_indx = X.shape[1] - 1 # index of ρ in β array (see below)

# stack t column for deterministic δt term (optional)
if d_trend == True:
    X      = np.hstack([X,ct])
    δ_indx = X.shape[1] - 1 # index of δ in β array (see below)


# --------------------- Perform linear regression; β = (X'X)^{-1}Xy:
y = y1[p+1:t_max+1,0] 
β,XXinv = regress(X,y)
print(f'\nregression coefficients : \n{β}')


#-------------------- construct estimated regression
j  = -p ; i = 0  ; y_est = np.matrix(np.zeros(t_max-p)).transpose()
while j <= p:
    Δyj   += lag1_diff(y2, t_max+j)[p+j:,0] 
    y_est += β[i,0]*Δyj
    j     += 1
α  =  β[α_indx ,0]
ρ  =  β[ρ_indx ,0]
y_est += α + ρ*yt
if d_trend == True:
    δ      =  β[δ_indx,0]
    y_est +=  δ*ct


#---------------------- OLS t statistic
# calculate s**2 (c.f. 17.4.30)
nβ  = β.shape[0]  # number of estimated coefficients  
ut  = y - y_est           ; ut2 = np.multiply(ut,ut)
s2  = np.sum(ut2)/(T-nβ)  ; s   = np.sqrt(s2)
print(f'\ns**2  = ({np.round(s,5)})**2')

# if no time trend (δt), calculate t_OLS for ρ = 1:
if d_trend == False:
    # calculate σ
    σ2 = s2*XXinv[ρ_indx, ρ_indx]
    σ  = np.sqrt(σ2)
    print(f'σ     = {np.round(σ,5)}')
    
    # OLS t-statistic
    t_OLS = (ρ-1)/σ
    print(f't_OLS = {np.round(t_OLS,5)}')
    


# if time trend (δt) included, calculate t_OLS for δ = 0:
if d_trend==True:
    σ2 = s2*XXinv[δ_indx, δ_indx]
    σ  = np.sqrt(σ2)
    print(f'σ(δ)    = {np.round(σ,5)}')
    
    # OLS t-statistic
    t_OLS = (δ-0)/σ
    print(f't_OLS for a [1,-1]:\n\
          t = {np.round(t_OLS,5)}')
    
    
# *NOTE! no t- test here! t_OLS is here only used to calculate testing 
# parameters in residual regression test.








#============================================
#================= Residual Test ============
#============================================
print(f'\n----------------------\nresidual test:')




# construct u_t = β2*u_(t-2) + β1*u_(t-1) + α regression matrix.
q = 2
j = 1  ;  
U = ut[q-j:cT-j,0] ; j += 1
while j <= 2:
    ut_j = ut[q-j:cT-j,0] ; 
    U    = np.hstack([U,ut_j])
    j   += 1

u = ut[q:,0] # don't cut ut yet!!

# regression
βu,UUinv = regress(U,u) ; nβu = βu.shape[0]
print(f'\nregression coefficients : \n{βu}')
β1,β2 = np.array(βu)[:,0]


    
# estimated regression of ut 
j = 1
u_est = 0
while j <=q:
    u_est += βu[j-1,0]*ut[q-j:cT-j,0] 
    j += 1
print(f'\nU: \n{U[:7,:]}')
ut = ut[q:,0] # now cut ut
print(f'\nut: \n{ut[:7,0]}')
# error on regression of ut
et = ut - u_est ; et2 = np.multiply(et,et)
# note that σ1 maybe should be labelled s1 instead.
σ1_sqrd = np.sum(et2)/(T-nβu)  ; σ1 = np.sqrt(σ1_sqrd)
print(f'\nσ1**2  = {np.round(σ1_sqrd,5)}')

#σ1 = np.sqrt(0.34395)
#β2 = 0.1292
#β1 = 0.6872
# calculate λ11 # c.f. 19.3.27
λ11 =  σ1/(1-β2-β1)
print(f'λ11    = {np.round(λ11,5)}')

# test a = [1,-1] c.f. pg. 611
tu = t_OLS*s/λ11
print(f'tu     = {np.round(tu,5)}')


if d_trend == False:
    # peform two sided t-test:
    tc = 1.96
    if abs(tu)<tc:
        print('\nAccept  Ho: a = [1,-1]')
    if abs(tu)>=tc:
        print('\nReject  Ho: a = [1,-1]')


if d_trend == True:
    # peform two sided t-test:
    tc = 1.96
    if abs(tu)<tc:
        print('\nAccept  Ho: no time trend')
    if abs(tu)>=tc:
        print('\nReject  Ho: no time trend')       
        
#============================================
#================= Plotting =================
#============================================

# plot CPI data
plt.figure() 
plt.plot(t_arr,y1,'b',t_arr,y2,'r' )
plt.title("blue (ct):  real personal consumption income expend. (US)\n\
          red   (yt) : real personal disposable income (US)        ")


# plot regression:  ct = Σς_j Δyj + α + γy_t
#*Note! confusingly, ct array here stands for 'cut time' array. But the example
# being followed happens to use 'ct' for what we are calling y = y1 = yt1
plt.figure() 
plt.plot(ct,y,'r',ct,y_est,'b--')
plt.title("blue--:ct (consumption expend.) \nregressed on yt (disposable income):\n\
          ct = Σς_j Δyj + α + γy_t\n\
          red   : actual ct data.")
 























