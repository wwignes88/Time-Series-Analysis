import numpy as np

# input for kargs should be γ,σ,φ,θ = kargs

def innovations(x, θ, φ, σ, kargs, θk):

    p = φ.shape[1]-1 ; q = θ.shape[1]-1 ; m = np.max([p,q])
    
    globals()['v0']  = Kij(1,1,kargs) ; 
    v  = np.matrix([np.round(v0,4)])
    N  = x.shape[1] ; n = 1
    #input(f'θk,N = {θk,N}')
    Mθ = np.zeros([N+1,θk]) 
    
    while n <= N:
        globals()[f'θ{n}0'] = 1
        k = 0
        while k < n:
            vk     = globals()[f'v{k}']
            θnn_k  = Kij(n+1,k+1,kargs) 
            j = 0
            while j <= k-1:
                #print('*here')
                θnn_j  = globals()[f'θ{n}{n-j}']
                θkk_j  = globals()[f'θ{k}{k-j}']
                vj     = globals()[f'v{j}']
                #print(f'   θ{n}{n-j} = {θnn_j}')
                #print(f'   θ{k}{k-j} = {θkk_j}')
                #print(f'   v{j} = {vj}')
                θnn_k += -θnn_j*θkk_j*vj
                j += 1
            θnn_k = θnn_k/vk
            globals()[f'θ{n}{n-k}'] = θnn_k
            
            i = 0
            while i < θk:
                if n-k == i+1 :
                    Mθ[n,i] = np.round(θnn_k,4)
                i += 1
            k += 1


        #----------------------------------
        vn   = Kij(n+1,n+1,kargs)
        j = 0
        while j <= n-1:
            θnn_j  = globals()[f'θ{n}{n-j}']
            vj     = globals()[f'v{j}']
            vn    += - vj*θnn_j**2
            j     += 1
        v    = np.append(v,[[np.round(vn,4)]],axis = 1)
        globals()[f'v{n}'] = vn
        n += 1
    Mθ = np.hstack([v.transpose(),Mθ])
    
    
    
    #---------------------------------------
    # find x^_{n+1} predictors (projections)
    xhat = np.matrix('0')
    globals()['xhat1']  = 0
    n   = 1 ; N_arr = np.matrix([0])
    while n <= N:
        N_arr = np.append(N_arr,[[n]],axis = 1)
        if n <= p:
            xhatn1 = 0
            j = 1
            while j <= n:
                xhatn1_j = globals()[f'xhat{n+1-j}']
                θnj      = globals()[f'θ{n}{j}']
                xhatn1  += θnj*(x[0,n-j] - xhatn1_j)
                j += 1
            globals()[f'xhat{n+1}'] = xhatn1
            xhat = np.append(xhat,[[np.round(xhatn1,3)]],axis = 1)
        
        if n > p:
            xhatn1 = 0
            l = 1
            while l <= p:
                xhatn1 += φ[0,l]*x[0,n-l]
                l += 1
            
            j = 1
            while j <= q:
                xhatn1_j = globals()[f'xhat{n+1-j}']
                θnj      = globals()[f'θ{n}{j}']
                xhatn1  += θnj*(x[0,n-j] - xhatn1_j)
                print(f'x_{n+1} = {xhatn1}')
                j += 1
            globals()[f'xhat{n+1}'] = xhatn1
            if n < N-1:
                o = 0
                #print(f'x{n+1}  = {np.round(x[0,n+1],3)}')
            #print(f'   xhat{n+1} = {np.round(xhatn1,3)}')
            xhat = np.append(xhat,[[np.round(xhatn1,3)]],axis = 1)
         
        n    += 1
    x = np.append(x,[[0]],axis = 1)  # append one val. to make length match.

    Mθ = np.hstack([x.transpose(),Mθ])
    Mθ = np.hstack([N_arr.transpose(),Mθ])
    Mθ = np.hstack([Mθ,xhat.transpose()])
    
    return Mθ

#θ = -0.9 ; σ = 1
#x  = np.matrix('-2.58 1.62 -0.96 2.62 -1.36')
#innovations(x, θ, σ) 


#----------------------------------------------------
# c.f. p. 168 or table on 171
# this is the generalized value of kij = E[WiWj] for 
# ARMA(p,q) process.
# args = γ_input,σ,φ,θ where γ_input is a string (first or second method)
# or a dictionary (third method).
def Kij(j,i,args):
    γ,σ,φ,θ = args

 
    q = θ.shape[1]-1  ; p = φ.shape[1]-1
    m = np.max([p,q])
    if i >=1 and j <= m:
        kij = γ(i-j)/σ**2
        #print(f'k{j}{i} = γ{abs(i-j)}  = {kij}')
        return kij
    
    if np.min([i,j]) <= m < np.max([i,j]) <= 2*m:
        kij = γ(i-j)
        r = 1
        while r <= p:
            kij += -φ[0,r]*γ(r-abs(i-j))
            r   += 1
        kij = kij/σ**2
        #print(f'k{j}{i}  = γ{abs(i-j)} -Σφγ = {kij}')
        return kij
    if np.min([i,j])>m:
        kij = 0
        r   = 0 
        while r <= q:
            if r+abs(i-j) > q:
                kij += 0
            if r+abs(i-j) <= q:
                kij += θ[0,r]*θ[0,r+abs(i-j)]
            
            r   += 1
        #print(f'k{j}{i}  = Σθθ = {kij} ')
        return kij
    else:
        #print(f'k{i}{j} = 0')
        kij = 0

    return kij









