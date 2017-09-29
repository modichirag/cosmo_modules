"""
This solves the particle mesh given intial density field, time steps and cosmology.

Additional options to choose smoothening and use 2LPT initial conditions
Use kick-drift-kick for evolution
In acceleration, poisson is solved by (/k**2) and derivative with (i*k)
"""


import numpy
import math
import cosmology
import evolve
import tools


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def fourier_lap(pm):
    #pm.r2c()
    pm.transfer([tools.pmLaplace])
    #pm.c2r()

def fourier_lap_finite(pm):
    #pm.r2c()
    pm.transfer([tools.pmLaplace_finite])
    #pm.c2r()

def fourier_der(pm, d):
    pm.r2c()
    pm.transfer([tools.pmDiff(d)])
    pm.c2r()

def lanzcos_der(pm, d):
    pm.r2c()
    pm.transfer([tools.pmLanzcos_Diff])
    pm.c2r()

def twopoint_der(pm, d):
    tools.twopt_Diff(pm, d)

def fourpoint_der(pm, d):
    tools.fourpt_Diff(pm, d)


###### This one works -- DO NOT TOUCH #######

def accelerate(pos, pmfinal, acc, smoothing, weight, laplacian, derivative):
    
    boxsize = pmfinal.BoxSize[0]
    nmesh = pmfinal.Nmesh
    
    pmfinal.clear()
    
    pmfinal.paint(pos, mass = weight)
    mean = pmfinal.real.mean()
    pmfinal.real -= mean
    
    pmfinal.r2c()

    pmfinal.transfer([tools.pmGauss(smoothing)])
#    pmfinal.transfer([laplacian])
#    laplacian(pmfinal)
#    pmfinal.complex *= -1.
    pmfinal.transfer([green])
    pmfinal.c2r()

    for d in range(3):
        
        pmfinal.push()

        # This commented portion takes derivative by rolling the array
        # Current implementation multiplies by i*k in fourier space

#         temp1 = numpy.roll(pmfinal.real, -1, axis = d)
#         temp2 = numpy.roll(pmfinal.real, 1, axis = d)
# #         g = -  (temp1 - temp2)/(2.*boxsize/nmesh)
#         temp3 = numpy.roll(pmfinal.real, -2, axis = d)
#         temp4 = numpy.roll(pmfinal.real, 2, axis = d)
#         g = - (2*(temp1 - temp2)/3. - (temp3 - temp4)/12.)/(boxsize/nmesh)
#         g *= 4*math.pi*43.007
#         pmfinal.clear()
#         pmfinal.real[:] = g

#        pmfinal.r2c()
        derivative(pmfinal, d)
#        pmfinal.transfer([derivative(d)])
#        pmfinal.c2r()

        pmfinal.real *= -4*math.pi*43.007        
        acc[:, d] = pmfinal.readout(pos)
        
        pmfinal.pop()
        
def green(pmfinal, complex):
        
    sink = 0
    for ki in pmfinal.k:
        sink = sink + ki**2
    sink[0,0,0] = 1.
   
    complex /= -sink


def kick(vel, acc, loga1, loga2, cosmo):
    
    N = 1025
    points = numpy.linspace(loga1, loga2, N, endpoint = True)
    aval = numpy.exp(points)
    hval = cosmo.Ha(aval)
    dt_kick = numpy.trapz(1/(aval * hval), points)
    vel += acc*dt_kick

def drift(pos, vel, loga1, loga2, boxsize, cosmo):
    
    N = 1025
    points = numpy.linspace(loga1, loga2, N, endpoint = True)
    aval = numpy.exp(points)
    hval = cosmo.Ha(aval)
    dt_drift = numpy.trapz(1/(aval * aval * hval), points)
    pos += vel*dt_drift
    
    pos[ pos < 0.0] += boxsize
    pos[ pos > boxsize] -= boxsize
    

def za_ic(pm, pmfinal, Q, timesteps, use_cosmo, smoothing = 0):
    
    cosmo = cosmology.Cosmology(use_cosmo, 100.)

    ZA = evolve.ZA(pm, Q)
    a0 = numpy.exp(timesteps[0])
    pos_ic = Q + ZA*cosmo.Dgrow(a0)
        
    pos_ic[ pos_ic < 0.0] += pm.BoxSize[0]
    pos_ic[ pos_ic > pm.BoxSize[0]] -= pm.BoxSize[0]
    
    pmfinal.clear()
    pmfinal.paint(pos_ic)


def lpt_ic(pm, pmfinal, Q, timesteps, use_cosmo, smoothing = 0):
    
    cosmo = cosmology.Cosmology(use_cosmo, 100.)

    ZA = evolve.ZA(pm, Q)
    LPTd = evolve.LPT2(pm, Q)
    a0 = numpy.exp(timesteps[0])
    pos_ic = Q + ZA*cosmo.Dgrow(a0)
    pos_ic += cosmo.Dgrow(a0)**2 *LPTd
          
    pos_ic[ pos_ic < 0.0] += pm.BoxSize[0]
    pos_ic[ pos_ic > pm.BoxSize[0]] -= pm.BoxSize[0]

    pmfinal.clear()
    pmfinal.paint(pos_ic)


def EvolvePmesh(pm, pmfinal, timesteps, Q, use_cosmo, smoothing = 0, lptbool = 0, snaps = None, \
                    laplacian = fourier_lap, derivative = fourier_der, zolamode = 0):
    
    output = []
    boxsize = pm.BoxSize[0]
    
    ZA = evolve.ZA(pm, Q)
    if lptbool:
        LPT = evolve.LPT2(pm, Q)

    obj1 = Bunch()
    obj1.pos = numpy.zeros_like(Q)
    obj1.vel = numpy.zeros_like(Q)
    obj1.accel = numpy.zeros_like(Q)

    ## IC ##
    cosmo = cosmology.Cosmology(use_cosmo, 100.)

    a0 = numpy.exp(timesteps[0])
    obj1.pos = Q + ZA*cosmo.Dgrow(a0)
    obj1.vel = ZA*cosmo.Dgrow(a0)*cosmo.Fomega1(a0)*cosmo.Ha(a0)* a0**2

    if lptbool:
        obj1.pos += cosmo.Dgrow(a0)**2 *LPT
        obj1.vel += cosmo.Dgrow(a0)**2 *LPT \
            *cosmo.Fomega2(a0)*cosmo.Ha(a0)* a0**2
        
    obj1.pos[ obj1.pos < 0.0] += pm.BoxSize[0]
    obj1.pos[ obj1.pos > pm.BoxSize[0]] -= pm.BoxSize[0]

    ## Loop it ##
    for i in range(len(timesteps) - 1):

        loga1 = timesteps[i]
        loga2 = timesteps[i+1]

        weight = 3*cosmo.M*cosmo.H0**2 /(8 * math.pi* 43.007) * (pm.BoxSize[0]/pm.Nmesh)**3 

        accelerate(obj1.pos, pmfinal, obj1.accel, smoothing, weight, laplacian, derivative)

        if i > 0:
            loga0 = timesteps[i-1]
            kick(obj1.vel, obj1.accel, 0.5*(loga1 + loga0), loga1 , cosmo)

        kick(obj1.vel, obj1.accel, loga1, 0.5*(loga1 + loga2), cosmo)

        drift(obj1.pos, obj1.vel,  loga1, loga2, boxsize, cosmo)
    
        pmfinal.clear()
        # if snaps != None:
       #     postemp = numpy.zeros_like(obj1.pos)
       #     postemp[:] = obj1.pos
       #     
       #     left = snaps.searchsorted(loga1, side = "left")
       #     right = snaps.searchsorted(loga2,  side = "right")
       #
       #     if left != right:
       #         drift(postemp, obj1.vel, loga1, snaps[left], boxsize, cosmo)
       #         pmfinal.clear()
       #         pmfinal.paint(postemp)
       #         output.append(pmfinal.real.copy())
       #
       #         for foo in range(left+1, right):
       #             drift(postemp, obj1.vel, snaps[foo -1], snaps[foo], boxsize, cosmo)
       #             pmfinal.clear()
       #             pmfinal.paint(postemp)
       #             output.append(pmfinal.real.copy())
       #             
                #drift(postemp, obj1.vel, snaps[right-1], loga2, pmfinal, cosmo)
                
       

    pmfinal.clear()
    ## Return evolved position ##
    pmfinal.paint(obj1.pos)
    
    return output



#class zola():
    
