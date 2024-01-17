import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time

ngrid = 360
lbox = 2000

niter = 200
correct_power = True

filename = 'dm_field_abacus_rankorder.DAT'
target_filename = 'dm_field_random1_tetcorr.DAT'

outfilename = 'out_field.dat'

# WARNING:
# The correct_power option makes sense only in the case in which niter>1.
# On the other hand, niter>1 with correct_power=False makes no sense.
# Hence:
# niter=1: it performs only one rank ordering.
#          If correct_power=False, it doesn't compensate for the error introduced in the P(k)
#          If correct_power=True, it applies a kernel to compensate the error in the P(k)
# niter>1: set correct_power=True 

# ***********************************************
# ***********************************************
@njit(parallel=True, cache=True, fastmath=True)
def rankorder(ngrid, amp11, amp22copy, amp11ind):

    for ii in prange(ngrid**3):

        amp11[amp11ind[ii]] = amp22copy[ii]

    return amp11

def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************                             
@njit(cache=True, fastmath=True)
def k_squared(lbox,nc,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= nc/2:
        kx = kfac*ii
      else:
        kx = -kfac*(nc-ii)

      if jj <= nc/2:
        ky = kfac*jj
      else:
        ky = -kfac*(nc-jj)

      #if kk <= nc/2:                                                                                                                                   
      kz = kfac*kk
      #else:                                                                                                                                            
      #  kz = -kfac*np.float64(nc-k)                                                                                                                    
      k2 = kx**2+ky**2+kz**2

      return k2

@njit(cache=True, fastmath=True)
def k_squared_nohermite(lbox,nc,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= nc/2:
        kx = kfac*ii
      else:
        kx = -kfac*(nc-ii)

      if jj <= nc/2:
        ky = kfac*jj
      else:
        ky = -kfac*(nc-jj)

      if kk <= nc/2:
          kz = kfac*kk
      else:
          kz = -kfac*(nc-kk)

      k2 = kx**2+ky**2+kz**2

      return k2

# **********************************************

def convolve_spectrum(signal, nc, kk, pk):
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)
    
    kmax = np.pi * nc / lbox * np.sqrt(3) #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))                                                                      

    fsignal = convolve_power(fsignal, kmax, nc, kk, pk)
    
    signal = np.fft.ifftn(fsignal) #* np.sqrt(nc**6/lbox**3)                                                                                            
    signal = signal.real

    return signal
  
# ********************************************** 
@njit(parallel=False, cache=True, fastmath=True)
def convolve_power(fsignal, kmax, nc, kk, pk):

    for i in prange(nc):
        for j in prange(nc):
            for k in prange(nc):

                ktot = np.sqrt(k_squared_nohermite(lbox,nc,i,j,k))
                kern = np.interp(ktot, kk,pk)
                fsignal[i,j,k] = fsignal[i,j,k] * kern

    return fsignal
  
# **********************************************                                                                          
def measure_spectrum(signal, nc):

    nbin = round(nc/2)

    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)                                                                                                  

    kmax = np.pi * nc / lbox * np.sqrt(3) #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))                                                                      
    dk = kmax/nbin  # Bin width                                                                                                                         

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode, nc)

    return kmode[1:], power[1:], nmode[1:]

# ********************************************** 

@njit(parallel=False, cache=True, fastmath=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode, nc):

    for i in prange(nc):
        for j in prange(nc):
            for k in prange(nc):
                ktot = np.sqrt(k_squared_nohermite(lbox,nc,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    norm = lbox**3/nc**6
    power *= norm

    return kmode, power, nmode



# ***********************************************                                             
# ***********************************************

# Import fields

if niter<=1:
    correct_power = False

raw1 = np.fromfile(filename, dtype=np.float32)
raw1 = np.reshape(raw1, (ngrid,ngrid,ngrid))

raw2 = np.fromfile(target_filename, dtype=np.float32)
raw2 = np.reshape(raw2, (ngrid,ngrid,ngrid))

kkt, pkt, nmodet = measure_spectrum(raw2, ngrid)
kk1, pk1, nmode1 = measure_spectrum(raw1, ngrid)

kern = np.ones(len(pkt))

# Prepare the fields
raw1 = raw1.flatten()

raw2 = raw2.flatten()

raw2copy = raw2.copy()
raw2copy = np.sort(raw2copy)

# Enter loop. What is done is:
# 1) convolve with the kernel (skip for first iteration)
# 2) do rank  ordering to the target field
# 3) measure new P(k)
# 4) update kernel

raw1new = raw1.copy()

print('')
print('Starting iterative rank ordering ...')
print('')

itlist = []
pkchisqlist = [] 

for ii in range(niter):

    print('--> ITERATION %d' %(ii+1))
    print('')

    if ii>0:
        print('Convolving with kernel ...')
        raw1new = np.reshape(raw1new, (ngrid,ngrid,ngrid))
        raw1new = convolve_spectrum(raw1new, ngrid, kkt, kern)
        raw1new = raw1new.flatten()

    #print(raw1new)

    raw1ind = np.argsort(raw1new)

    raw1ind = np.argsort(raw1new)

    print('Applyng rank ordering ...')
    raw1new = rankorder(ngrid, raw1new, raw2copy, raw1ind)

    print('Measuring new P(k) and computing kernel ...')
    raw1new = np.reshape(raw1new, (ngrid,ngrid,ngrid))
    kktemp, pktemp, nmodetemp = measure_spectrum(raw1new, ngrid)
    raw1ew = raw1new.flatten()

    #print(pkt/pktemp)
    kerntemp = np.sqrt(pkt / pktemp)

    if correct_power == True:
        kern *= kerntemp

    pkchisq = np.sum((pktemp-pkt)**2)

    print('P(k) chi square: ', pkchisq)

    itlist.append(ii)
    pkchisqlist.append(pkchisq)

    print('... done!')
    print('')
    print('')

raw1new.astype('float32').tofile(outfilename)

f = open('kernel_interative.txt', 'w')

for jj in range(len(kern)):
    f.write(str(kkt[jj]) + '      ' + str(kern[jj]) + '\n')
f.close()

g = open('chisquare.txt', 'w')

for jj in range(len(itlist)):
    g.write(str(itlist[jj]) + '      ' + str(pkchisqlist[jj]) + '\n')
g.close()
    
tf = time.time()
print('Elapsed %s hours ...' %str((tf-ti)/3600.))
