"""
This module moves the particles on the mesh. 

You can output ZA or 2 LPT displacement \
and shift the particles by ZA (+2LPT) displacement. Displacements are not multiplied by\
growth functions yet.
"""

import tools
import numpy


def ZAdrift(pminitial, pmfinal, pos, smoothing = 0, deconvolve = 0, lptbool = 0):

    """
    This function takes in initial density field, calculates ZA displacement, \
    shifts the particle with that displacement and paints them. 
    """
    if not lptbool:
        ZAdisp = ZA(pminitial, pos)
    elif lptbool == 1:
        ZAdisp = ZA(pminitial, pos)
        ZAdisp += LPT2(pminitial, pos)
    elif lptbool == 2:
        ZAdisp = LPT2(pminitial, pos)

    pmfinal.clear()
    pmfinal.paint(pos + ZAdisp)
    
    pmfinal.r2c()
    if smoothing:
        pmfinal.transfer([tools.pmGauss(smoothing)])
    if deconvolve:
        pmfinal.transfer([tools.pmWcic(deconvolve)])
    pmfinal.c2r()

    return ZAdisp



def ZA(pm, pos):

    """
    This function takes in initial density field and positions
    and returns the ZA displacement. 
    """
    pm.push()
    ZAdisp = numpy.zeros_like(pos)
    pm.r2c()
    pm.transfer([tools.pmLaplace]) 
    for dir in range(3):
        pm.push()
        pm.transfer([tools.pmDiff(dir)])
        pm.c2r()
        tmp = pm.readout(pos)
        ZAdisp[:, dir] = tmp
        pm.pop()
    pm.pop()

    return ZAdisp



def LPT2(pm, Q):
    
    """
    This function takes in initial density field and positions
    and returns the 2LPT displacement. 
    """
    pm.push()
    
    pm.r2c()
    pm.transfer([tools.pmLaplace])
    phiq = []
    for i in range(3):
        for j in range(i,3):
            pm.push()
            pm.complex *= pm.k[i]*pm.k[j]
            pm.c2r()
            phiq.append(pm.real.copy())
            pm.pop()
    
    fq = phiq[0]* phiq[3] + phiq[0]* phiq[5] + phiq[3]* phiq[5] - phiq[1]**2 - phiq[2]**2 - phiq[4]**2
    
    pm.clear()
    pm.real[:] = fq
    
    LPT = ZA(pm, Q)*3/7.
    
    pm.pop()
    
    return LPT


def LPT3(pm, Q):
    pass
