import numpy as np
#import matplotlib.pyplot as plt
from numba import njit, prange
import time
import scipy.optimize
import scipy.ndimage

ngrid = 256
lbox = 1000

ntarget = 1024

niter = 1
correct_power = False

filename = 'dm_field_abacus_rankorder.DAT'
target_filename = 'dm_field_random1_tetcorr.DAT'

outfilename = 'out_field.dat'

apply_nl_transform = True
calibrate_nl_transform = True
base_nl = 1.35

# WARNING:
# The correct_power option makes sense only in the case in which niter>1.
# On the other hand, niter>1 with correct_power=False makes no sense.
# Hence:
# niter=1: it performs only one rank ordering.
#          If correct_power=False, it doesn't compensate for the error introduced in the P(k)
#          If correct_power=True, it applies a kernel to compensate the error in the P(k)
# niter>1: set correct_power=True 

# Use either iterative rank ordering, or NL transform. At the moment, if both options are enabled, the code throws an error ad exits.

seed = 123456 # Arbitrary number, but once you pick one, leave it fixed

prec_dm = 'float32'
prec_dm_target = 'float32'

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

# Set the seed for reproducibility
np.random.seed(seed)

# Make initial check
if correct_power==True and apply_nl_transform==True:
    print('Error: terative power correction and application of a NL transform are incompatible.')
    print('Switch on only one of the two. Exiting.')
    exit()

# Import fields

if niter>1:
    correct_power = True

if prec_dm=='float32':
    dtype_dm=np.float32
elif prec_dm=='float64':
    dtype_dm=np.float64
else:
    print('Error: precision type not found. Exiting.')

if prec_dm_target=='float32':
    dtype_dm_target=np.float32
elif prec_dm_target=='float64':
    dtype_dm_target=np.float64
else:
    print('Error: precision type not found. Exiting.')

raw1 = np.fromfile(filename, dtype=dtype_dm)
raw1 = np.reshape(raw1, (ngrid,ngrid,ngrid))

raw2 = np.fromfile(target_filename, dtype=dtype_dm_target)
raw2 = np.reshape(raw2, (ntarget,ntarget,ntarget))


if ntarget>ngrid:

    print('Found ntarget>ngrid. Extracting a random subset of cells from the full target field.')

    dummyrange = np.arange(ntarget**3)
    np.random.shuffle(dummyrange)
    dummy = dummyrange[:ngrid**3]

    raw2= raw2.flatten()

    raw2 = raw2[dummy]

    raw2 = np.reshape(raw2, (ngrid,ngrid,ngrid))
    
elif ntarget<ngrid:

    print('Found ntarget<ngrid. This case is not implemented. Set ntarget>=ngrid. Exiting')
    exit()

else:
    print('Found ntarget=ngrid. All good, keep on going.')

    
# Measure power spectra    
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

    if apply_nl_transform == True:

        if calibrate_nl_transform==True:

            def chisquare(xbase, raw1dummy):

                raw1newdummy = xbase**raw1dummy/np.mean(xbase**raw1dummy) - 1.

                raw2smooth = scipy.ndimage.gaussian_filter(raw2, sigma=60/(lbox/ngrid), mode='wrap')
                vartarget = np.std(raw2smooth)

                raw1newsmooth = scipy.ndimage.gaussian_filter(raw1newdummy, sigma=60/(lbox/ngrid), mode='wrap')
                vartmp = np.std(raw1newsmooth)

                chisq = (vartarget-vartmp)**2

                return chisq

            res = scipy.optimize.minimize(chisquare, [1.], args=(raw1new))
            base = res[0]

        else:
            base = base_nl
            
        raw1new = base**raw1new/np.mean(base**raw1new) - 1.


    print('Measuring new P(k) and computing kernel ...')
    raw1new = np.reshape(raw1new, (ngrid,ngrid,ngrid))
    kktemp, pktemp, nmodetemp = measure_spectrum(raw1new, ngrid)
    raw1new = raw1new.flatten()

    #print(pkt/pktemp)
    kerntemp = np.sqrt(pkt / pktemp)

    if correct_power == True:
        kern *= kerntemp

    pkchisq = np.sum((pktemp-pkt)**2)

    print('P(k) chi square: ', pkchisq)
    print('P(k) large-scale bias: ', np.mean(pktemp[:10]/pkt[:,10]))

    itlist.append(ii)
    pkchisqlist.append(pkchisq)

    print('... done!')
    print('')
    print('')

raw1new.astype(prec_dm).tofile(outfilename)

if niter>1 and correct_power==True:
    
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
