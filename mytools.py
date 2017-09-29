"""
This module has useful functions like smoothing, window-deconvolve, derivative etc.

Transfer functions associated with ParticleMesh objects are named pmXyz where Xyz 
is the logical name.

"""

import numpy
import math

from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.integrate import simps
from scipy.misc import derivative

def pmLaplace(pm, complex):
    """
    Divide the complex part by k**2 (norm squared)
    """
    k2 = 0
    for ki in pm.k:
        k2 = k2 + ki ** 2
    k2[k2 == 0] = 1.0
    complex /= k2

def pmLaplace_finite(pm, complex):
    """
    Divide the complex part by 4*sin(k_i/2)**2 summed
    """
    k2 = 0
    for ki in pm.k:
        k2 = k2 + 4*numpy.sin(ki/2.)**2
    k2[k2 == 0] = 1.0
    complex /= k2


def pmDiff(dir):
    """
    Multiply the complex part by i*k(dir)
    """
    def pmDiff(pm, complex):
        complex *= pm.k[dir] * 1j
    return pmDiff

def pmLanzcos_Diff(dir):
    """
    Multiply the complex part by i*k(dir)
    """
    def pmLanzcos_Diff(pm, complex):
        wi = pm.k[dir]*pm.BoxSize[0]/pm.Nmesh
        tmp = 1 / 6.0 * (8 * numpy.sin (wi) - numpy.sin (2 * wi))
        complex *= tmp * 1j * pm.Nmesh/pm.BoxSize[0]
    return pmLanzcos_Diff

def twopt_Diff(pm, dir):
    """
    Multiply the complex part by i*k(dir)
    """
    temp1 = numpy.roll(pm.real, -1, axis = dir)
    temp2 = numpy.roll(pm.real, 1, axis = dir)
    g =  (temp1 - temp2)/(2.*pm.BoxSize[0]/pm.Nmesh)    
    pm.real[:] = g 

def fourpt_Diff(pm, dir):
    """
    Multiply the complex part by i*k(dir)
    """
    temp1 = numpy.roll(pm.real, -1, axis = dir)
    temp2 = numpy.roll(pm.real, 1, axis = dir)
    temp3 = numpy.roll(pm.real, -2, axis = dir)
    temp4 = numpy.roll(pm.real, 2, axis = dir)
    g = (2*(temp1 - temp2)/3. - (temp3 - temp4)/12.)/(pm.BoxSize[0]/pm.Nmesh)
    g = -  (temp1 - temp2)/(2.*pm.BoxSize[0]/pm.Nmesh)    
    pm.real[:] = g 


def pmGauss(sm):
    '''Do Gauss smoothing for the particle mesh
    '''    
    def pmGauss(pm, complex):
        k2 = 0
        for ki in pm.k:
            k2 =  k2 + ki ** 2

        complex *= numpy.exp(-0.5*k2*(sm**2))
        #complex *= k2 < 0.25
    return pmGauss


def pmTophat(sm):
    '''Do Tophat smoothing for the particle mesh
    '''    
    def pmTophat(pm, complex):
        k2 = 0
        for foo in pm.k:
            k2 = k2 + foo**2        
        kr = sm * k2**0.5
        kr[0, 0, 0] = 1
        wt = 3 * (numpy.sin(kr)/kr - numpy.cos(kr))/kr**2
        wt[0,0,0] = 1        
        complex *= wt

    return pmTophat


def pmFingauss(sm):
    '''Do finite Gaussian smoothing by solving diffusion equation
    '''
    def pmFingauss(pm, complex):
        kny = numpy.pi*pm.Nmesh/pm.BoxSize[0]

        k2 = 0
        for ki in pm.k:
            k2 =  k2 + ((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny))) ** 2

        complex *= numpy.exp(-0.5*k2*(sm**2))
        #complex *= k2 < 0.25
    return pmFingauss

def pmWcic(exponent =0):
    '''Deconvolve with grid window with exponent (2 for CIC)
    '''
    def pmWcic(pm, complex):
        win = numpy.sinc(pm.k[0]*pm.BoxSize[0]/(2*pm.Nmesh*math.pi))*\
          numpy.sinc(pm.k[1]*pm.BoxSize[1]/(2*pm.Nmesh*math.pi))*\
          numpy.sinc(pm.k[2]*pm.BoxSize[2]/(2*pm.Nmesh*math.pi))
        
        complex /= win**exponent
    
    return pmWcic



def shear(pm):
    '''Takes in a PMesh object in real space. Returns am array of shear'''
    s2 = numpy.zeros_like(pm.real)

    pm.r2c()
    k2 = 0
    for foo in pm.k:
        k2 = k2 + foo**2
    k2[0,0,0] =  1

    for i in range(3):
        for j in range(i, 3):
            pm.push()
            pm.complex *= (pm.k[i]*pm.k[j] / k2 - diracdelta(i, j)/3.)
            pm.c2r()
            s2 += pm.real**2
            if i != j:
                s2 += pm.real**2
            pm.pop()
    pm.c2r()
    return s2


def diracdelta(i, j):
    if i == j:
        return 1
    else:
        return 0


def tophat(k, R):
    '''Takes in k, R scalar to return tophat window for the tuple'''
    kr = k*R
    wt = 3 * (numpy.sin(kr)/kr - numpy.cos(kr))/kr**2
    if wt is 0:
        wt = 1
    return wt

def gauss(k, R):
    '''Takes in k, R scalar to return gauss window for the tuple'''
    kr = k*R
    wt = numpy.exp(-(kr**2) /2.)
    return wt


def fin_gauss(k, kny, R):
    '''Takes in k, R and kny to do Gaussian smoothing corresponding to finite grid with kny'''
    kf = numpy.sin(k*numpy.pi/kny/2.)*kny*2/numpy.pi
    return numpy.exp(-(kf**2 * R**2) /2.)


def sigmamix(pfile, j, a, b, r1 , r2 = None):
    """returns sigma corresponding to r1 and r2 radius for mixed window. 
    w1(a) corresponds to r1, w2(b) corresponds r2"""
    k, p = numpy.loadtxt(pfile, unpack = True)
    
    if r2 is None:
        r2 = r1
    
    if a == "T":
        w1 = tophat(k, r1)
    elif a == 'G':
        w1 = gauss(k, r1)
    else:
        print ("Window 1 should be either G or T")
        
    if b == "T":
        w2 = tophat(k, r2)
    elif b == 'G':
        w2 = gauss(k, r2)
    else:
        print ("Window 2 should be either G or T")
    return numpy.sqrt(simps(p * w1 * w2 * k**2 * k**(2*j), k)/2/math.pi**2)


def loginterp(x, y, yint = None, side = "both", lorder = 15, rorder = 15, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6):

    if yint is None:
        yint = interpolate(x, y, k = 5)

    if side == "both":
        side = "lr"
        l =lp
        r =rp
    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
    print(lneff, rneff)

    xl = numpy.logspace(-18, numpy.log10(x[l]), 10**6.)
    xr = numpy.logspace(numpy.log10(x[r]), 10., 10**6.)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = numpy.concatenate((xl, xint))
        yint = numpy.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = numpy.concatenate((xint, xr))
        yint = numpy.concatenate((yint, yr))
    yint2 = interpolate(xint, yint, k = 5)

    return yint2




################# To calculate power #################

def fftk(shape, boxsize, symmetric=True):
    """
    
    """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)
        
        k.append(kd)
    kk = sum([i ** 2 for i in k])
    for i in k:
        del i
    del k, kd, kdshape
    return kk ** 0.5

def power(f1, f2=None, boxsize=1.0, k = None):
    """
    Calculate power spectrum given density field in real space & boxsize.
    Divide by mean
    """
#    f1 = f1[::2, ::2, ::2]
    c1 = numpy.fft.rfftn(f1)
    c1 /= c1[0, 0, 0].real
    c1[0, 0, 0] = 0
    if f2 is not None:
        c2 = numpy.fft.rfftn(f2)
        c2 /= c2[0, 0, 0].real
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    if k is None:
        k = fftk(f1.shape, boxsize)
    H, edges = numpy.histogram(k.flat, weights=x.flat, bins=f1.shape[0]) 
    N, edges = numpy.histogram(k.flat, bins=edges)
    center= edges[1:] + edges[:-1]
    
    return 0.5 * center, H *boxsize**3 / N


def power2(f1, f2=None, boxsize=1.0, k = None):
    """                                                                 
    Calculate power spectrum given density field in real space & boxsize
    Do not divide by mean, but the field should have 0 mean
    """
#     f1 = f1[::2, ::2, ::2]
    c1 = numpy.fft.rfftn(f1)
    c1 /= f1.shape[0]**3
    c1[0, 0, 0] = 0
    if f2 is not None:
#         f2 = f2[::2, ::2, ::2]
        c2 = numpy.fft.rfftn(f2)
        c2 /= f2.shape[0]**3
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    if k is None:
        k = fftk(f1.shape, boxsize)
    H, edges = numpy.histogram(k.flat, weights=x.flat, bins=f1.shape[0])
    N, edges = numpy.histogram(k.flat, bins=edges)
    center= edges[1:] + edges[:-1]
    
    return 0.5 * center, H *boxsize **3 /N


def power_cheap(f1, f2=None, boxsize=1.0, average=True):
    """ stupid power spectrum calculator.
        f1 f2 must be density fields in configuration or fourier space.
        For convenience if f1 is strictly overdensity in fourier space,
        (zero mode is zero) the code still works.
        Does not work neccesarily if mean is close to but not exactly 0 
        in real space, safe side- add 1 if its underdensity. 
        Returns k, p or k, p * n, N if average is False
    """
    def tocomplex(f1):
        if f1.dtype.kind == 'c':
            return f1
        else:
            return numpy.fft.rfftn(f1)

    f1 = tocomplex(f1)
    if f1[0, 0, 0] != 0.0:
        f1 /= abs(f1[0, 0, 0])

    if f2 is None:
        f2 = f1

    if f2 is not f1:
        f2 = tocomplex(f2)
        if f2[0, 0, 0] != 0.0:
            f2 /= abs(f2[0, 0, 0])

    def fftk(shape, boxsize):
        k = []
        for d in range(len(shape)):
            kd = numpy.arange(shape[d])

            if d != len(shape) - 1:
                kd[kd > shape[d] // 2] -= shape[d] 
            else:
                kd = kd[:shape[d]]

            kdshape = numpy.ones(len(shape), dtype='int')
            kdshape[d] = len(kd)
            kd = kd.reshape(kdshape)

            k.append(kd)
        return k

    k = fftk(f1.shape, boxsize)
    
    def find_root(kk):
        solution = numpy.int64(numpy.sqrt(kk) - 2).clip(0)
        solution[solution < 0] = 0
        mask = (solution + 1) ** 2 < kk
        while(mask.any()):
            solution[mask] += 1
            mask = (solution + 1) ** 2 <= kk

        return solution

#    ksum = numpy.zeros(f1.shape[0] //2, 'f8')
#    wsum = numpy.zeros(f1.shape[0] //2, 'f8')
#    xsum = numpy.zeros(f1.shape[0] //2, 'f8')

    ksum = numpy.zeros(f1.shape[0], 'f8')
    wsum = numpy.zeros(f1.shape[0], 'f8')
    xsum = numpy.zeros(f1.shape[0], 'f8')
    
    for i in range(f1.shape[0]):
        kk = k[0][i] ** 2 + k[1] ** 2 + k[2] ** 2

        # remove unused dimension
        kk = kk[0]

        d = find_root(kk)

        w = numpy.ones(d.shape, dtype='f4') * 2
        w[..., 0] = 1
        w[..., -1] = 1

        xw = abs(f1[i] * f2[i].conjugate()) * w

        kw = kk ** 0.5 * 2 * numpy.pi / boxsize * w

#        ksum += numpy.bincount(d.flat, weights=kw.flat, minlength=f1.shape[0])[:f1.shape[0] // 2]
#        wsum += numpy.bincount(d.flat, weights=w.flat, minlength=f1.shape[0])[:f1.shape[0] // 2]
#        xsum += numpy.bincount(d.flat, weights=xw.flat, minlength=f1.shape[0])[:f1.shape[0] // 2]

        ksum += numpy.bincount(d.flat, weights=kw.flat, minlength=f1.shape[0])[:f1.shape[0] ]
        wsum += numpy.bincount(d.flat, weights=w.flat, minlength=f1.shape[0])[:f1.shape[0] ]
        xsum += numpy.bincount(d.flat, weights=xw.flat, minlength=f1.shape[0])[:f1.shape[0] ]
        
    wsum[wsum ==0] = 1
    center = ksum / wsum

    if not average:
        return center, xsum * boxsize**3, wsum
    else:
        return center, xsum / wsum * boxsize **3


################# scale-factor <-> redshift #################

def ztola(z):
    return( numpy.log(1./(z+1)) )

def ztoa(z):
    return( 1./(z+1))

def latoz(la):
    return(1./numpy.exp(la) - 1)

def atoz(a):
    return(1./a -1)



################# Array Manipulations #################


def subsize_complex(array, nci):
    '''Subsize the array to shape (nci, nci, nci/2). nci should be of the form nc/2^m'''
    ncs = array.shape[0]
    arraysub = numpy.zeros(([nci, nci, nci//2 + 1]), dtype= "complex64")
    arraysub[:nci//2 + 1, :nci//2 + 1, :nci//2 + 1]  = array[:nci//2 + 1, :nci//2 + 1, :nci//2 + 1]
    arraysub[nci//2 + 1:, :nci//2 + 1, :nci//2 + 1]  = array[ncs - nci//2 + 1:, :nci//2 + 1, :nci//2 + 1]
    arraysub[:nci//2 + 1, nci//2 + 1:, :nci//2 + 1]  = array[:nci//2 + 1, ncs - nci//2 + 1:, :nci//2 + 1]
    arraysub[nci//2 + 1:, nci//2 + 1:, :nci//2 + 1]  = array[ncs - nci//2 + 1:, ncs - nci//2 + 1:, :nci//2 + 1]
    return arraysub

def subsize_real(array, n = None):
    '''Subsize the array to shape (n/2, n/2, n/2). nci should be of the form nc/2^m'''
    nc = array.shape[0]
    if n == None:
        n = int(nc/2)
    fac = int(nc/n)
    arraysub = numpy.zeros([n, n, n])
    for foo in range(fac):
        for boo in range(fac):
            for shoo in range(fac):
#                 arraysub += array[foo::fac, boo::fac, shoo::fac]/float(fac**3)
                arraysub += array[foo::fac, boo::fac, shoo::fac]
    return arraysub



def dktoarray3d(pcomplex):

    """
    Convert the complex fft-style array to 1D array of independent elements
    """
    nmesh = pcomplex.shape[0]
    x = numpy.zeros(nmesh**3) 
    # Structure to save- 
    # - first the real and complex part of N*N*(N/2-1) elements in between 0 & Nyq plane
    # - then real and complex part of N*(N/2-1) elements in between 0 & Nyq row on 0 plane
    # - then real and complex part of N*(N/2-1) elements in between 0 & Nyq row on Ny plane
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on 0,0 row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on 0,Ny row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on Ny,0 row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on Ny,Ny row
    # - then 8 real numbers in order (0,0,0)(Ny,0,0)(0,Ny,0)(Ny,Ny,0)(0,0,Ny)(Ny,0,Ny)(0,Ny,Ny)(Ny,Ny,Ny) 

    index = 0
    count = nmesh*nmesh*(nmesh/2-1)
    x[index : index + count] = pcomplex[:, :, 1:nmesh/2].real.flatten()
    index += count
    count = nmesh*nmesh*(nmesh/2-1)
    x[index : index + count] = pcomplex[:, :, 1:nmesh/2].imag.flatten()
    index += count


    for foo in [0, nmesh/2]:

        count = nmesh* (nmesh/2 -1)
        x[index : index + count] = pcomplex[:, 1:nmesh/2, foo].real.flatten()
        index += count
        count = nmesh* (nmesh/2 -1)
        x[index : index + count] = pcomplex[:, 1:nmesh/2, foo].imag.flatten()
        index += count


    for foo in [0, nmesh/2]:
        for boo in [0, nmesh/2]:

            count = nmesh/2 -1
            x[index : index + count] = pcomplex[1:nmesh/2, foo, boo].real.flatten()
            index += count
            count = nmesh/2 -1
            x[index : index + count] = pcomplex[1:nmesh/2, foo, boo].imag.flatten()
            index += count


    for foo in [0, nmesh/2]:
        for boo in [0, nmesh/2]:
            for shoo in [0, nmesh/2]:

                count = 1
                x[index : index +count] = pcomplex[shoo, boo, foo].real.flatten()
                index += count

    return(x)



def arraytodk3d(x):

    """
    Convert back 1d array of independent elements to correct complex array fft-style
    """
    nmesh = round(x.shape[0]**(1/3.))
    y = numpy.zeros((nmesh, nmesh, nmesh/2+1), dtype = numpy.complex128)

    # Structure saved- 
    # - first the real and complex part of N*N*(N/2-1) elements in between 0 & Nyq plane
    # - then real and complex part of N*(N/2-1) elements in between 0 & Nyq row on 0 plane
    # - then real and complex part of N*(N/2-1) elements in between 0 & Nyq row on Ny plane
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on 0,0 row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on 0,Ny row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on Ny,0 row
    # - then real and complex part of (N/2-1) elements in between 0 & Nyq element on Ny,Ny row
    # - then 8 real numbers in order (0,0,0)(Ny,0,0)(0,Ny,0)(Ny,Ny,0)(0,0,Ny)(Ny,0,Ny)(0,Ny,Ny)(Ny,Ny,Ny) 

    index = 0
    count = nmesh*nmesh*(nmesh/2-1)
    y[:, :, 1:nmesh/2].real = x[index : index + count].reshape(nmesh, nmesh, nmesh/2 -1)
    index += count
    count = nmesh*nmesh*(nmesh/2-1)
    y[:, :, 1:nmesh/2].imag = x[index : index + count].reshape(nmesh, nmesh, nmesh/2 -1)
    index += count


    for foo in [0, nmesh/2]:
        count = nmesh* (nmesh/2 -1)
        y[:, 1:nmesh/2, foo].real = x[index : index + count].reshape(nmesh, nmesh/2 -1)
        y[nmesh - 1:0:-1, nmesh - 1:nmesh/2:-1, foo].real = y[1:, 1:nmesh/2, foo].real
        y[0, nmesh - 1:nmesh/2:-1, foo ].real = y[0, 1:nmesh/2, foo ].real 
        index += count

        count = nmesh* (nmesh/2 -1)
        y[:, 1:nmesh/2, foo].imag = x[index : index + count].reshape(nmesh, nmesh/2 -1)
        y[nmesh - 1:0:-1, nmesh - 1:nmesh/2:-1, foo].imag = -1*y[1:, 1:nmesh/2, foo].imag
        y[0, nmesh - 1:nmesh/2:-1, foo ].imag = -1*y[0, 1:nmesh/2, foo ].imag 
        index += count


    for foo in [0, nmesh/2]:
        for boo in [0, nmesh/2]:
            count = nmesh/2 -1
            y[1:nmesh/2, foo, boo].real = x[index : index + count].reshape(nmesh/2 -1)
            y[nmesh -1:nmesh/2:-1, foo, boo].real = y[1:nmesh/2, foo, boo].real 
            index += count

            count = nmesh/2 -1
            y[1:nmesh/2, foo, boo].imag = x[index : index + count].reshape(nmesh/2 -1)
            y[nmesh -1:nmesh/2:-1, foo, boo].imag = -1*y[1:nmesh/2, foo, boo].imag 
            index += count



    for foo in [0, nmesh/2]:
        for boo in [0, nmesh/2]:
            for shoo in [0, nmesh/2]:

                count = 1
                y[shoo, boo, foo] = x[index]
                index += count

    return(y)
