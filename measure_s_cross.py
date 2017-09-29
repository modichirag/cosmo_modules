import numpy
import math

from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.integrate import simps

import sys
sys.path.append("/global/homes/c/chmodi/Programs/Py_codes/modules/")
import cosmology as cosmo_lib

class s_cross():

    def __init__(self, power_file, M, L, R = 0., H0 = 100.):

        self.M = M
        self.L = L
        self.ktrue, self.ptrue = numpy.loadtxt(power_file, unpack = True)
        self.H0 = H0
        self.R = R

        self.rhoc =  3 * H0**2 /(8 * math.pi * 43.007)
        self.rhom = self.rhoc*M
        self.cosmo = cosmo_lib.Cosmology(M= M, L = L)
        self.masses = 10**numpy.arange(9, 18, 0.01)
        self.sigma = numpy.zeros_like(self.masses)
        self.calc_sigma()

    def calc_sigma(self):
        M = self.masses
        for foo in range(len(M)):
            self.sigma[foo] = self.sigmasq(M[foo])**0.5


    def tophat(self, k, R):    
        kr = k*R
        wt = 3 * (numpy.sin(kr)/kr - numpy.cos(kr))/kr**2
        if wt is 0:
            wt = 1
        return wt

    def gauss(self, k, R):    
        kr = k*R
        wt = numpy.exp(-kr**2 /2.)
        return wt

    def rtom(self, R):
        """returns Mass in 10**10 solar mass for smoothing scale in Mpc"""
        m = 4* math.pi*self.rhom * R**3 /3.
        return m*10**10

    def sm_scale(self, M):    
        """returns smoothing scale in Mpc for Mass in 10**10 solar mass"""
        rhobar = self.rhom * 10**10
        R3 = 3* M /(4 * math.pi * rhobar)
        return R3 ** (1/3.)

    def sigmasq(self, M):
        """returns sigma**2 corresponding to mass in 10**10 solar mass"""
        R = self.sm_scale(M)
        k = self.ktrue
        p = self.ptrue
        w1 = self.tophat(k, R)
        w2 = self.tophat(k, self.R)
        return simps(p * w1 * w2 * k**2, k)/2/math.pi**2


    def dlninvsigdM(self, sigmaf, M, a = 1.):
        """ returns d(ln(1/sigma))/d(M) for M in 10**10 solar masses. Can specify redshift with scale factor"""

        dM = 0.001 * M
        Mf = M + dM/2.
        Mb = M - dM/2.
        lnsigf = numpy.log(1/sigmaf(a)(Mf))
        lnsigb = numpy.log(1/sigmaf(a)(Mb))
        return (lnsigf - lnsigb)/dM

    def sigmaf(self, a = 1.):
        """ returns interpolating function for sigma. syntax to use - sigmaf(a)(M)"""
        d = self.cosmo.Dgrow(a=a)
        return interpolate(self.masses, d*self.sigma)


    def sigmamix(self, M, j, a, b, r1 = None, r2 = None):
        """returns sigma corresponding to mass in 10**10 solar mass for mixed window. 
        w1(a) corresponds to M, w2(b) corresponds to large scale R of the class"""
        if r1 is None:
            r1 = self.sm_scale(M)
        if r2 is None:
            r2 = self.R
        k = self.ktrue
        p = self.ptrue
        if a == "T":
            w1 = self.tophat(k, r1)
        elif a == 'G':
            w1 = self.gauss(k, r1)
        else:
            print ("Window should be either G or T")

        if b == "T":
            w2 = self.tophat(k, r2)
        elif b == 'G':
            w2 = self.gauss(k, r2)
        else:
            print ("Window should be either G or T")
        return numpy.sqrt(simps(p * w1 * w2 * k**2 * k**(2*j), k)/2/math.pi**2)
