def icdf_sampling(bs, mf = None, mv = None, mfv = None, match_high = True, hmass = None, M0 = None, N0 = None):
    '''
    Given samples from analytic mass function (dN/dln(M)), find halo masss by matching abundance via
    inverse cdf sampling. 
    bs : boxsize
    mv, mfv : (Semi-optional) analytic hmf sampled at masses 'mv'
    mf : (Semi-optional) Analytic hmf, if mv and mfv are not given
    match_high : if True, Match the highest mass of the catalog to analytic mass.
        if False, match the lowest mass
    hmass : (Semi-Optional) Halo mass catalog, used to calculate highest/lowest mass 
        and number of halos
    M0, N0 : (Semi-optional) If mass catalog not given, M0 and N0 are required to 
        correspond to highest/lowest mass and number if halos
        
    Returns: Abundance matched halo mass catalog
    '''
    Nint = 500 #No.of points to interpolate
    #Create interpolating function for mass_func
    if mf is not None:
        mv = np.logspace(10, 17, Nint)
        mfv = mf(mv)
    elif mv is None:
            print("Need either a function or values sampled from the analytic mass function to match against")
            return None
    #Interpolate
    imf = interpolate(mv, mfv, k = 5)
    ilmf = lambda x: imf(np.exp(x))
    
    #Create array to integrate high or low from the matched mass based on match_high
    if N0 is None:
        N0 = hmass.size
    if match_high:
        if M0 is None:
            if hmass is None:
                print("Need either a halo mass catalog or a mass to be matched at")
                return 0
            else:
                M0 = hmass.max()
        lmm = np.linspace(np.log(M0), np.log(mv.min()), Nint)
    else:
        if M0 is None:
            M0 = hmass.min()
        lmm = np.linspace(np.log(M0), np.log(mv.max()), Nint)

        
    #Calculate the other mass-limit M2 of the catalog by integrating mf and comparing total number of halos
    ncum = abs(np.array([romberg(ilmf, lmm[0], lmm[i]) for i in range(lmm.size)]))*bs**3
    M2 = np.exp(np.interp(N0, ncum, lmm))

    #Create pdf and cdf for N(M) from mf between M0 and M2
    lmm2 = np.linspace(np.log(M0), np.log(M2), Nint)
    nbin = abs(np.array([romberg(ilmf, lmm2[i], lmm2[i+1]) for i in range(lmm2.size-1)]))*bs**3
    nprob = nbin/nbin.sum()
    cdf = np.array([nprob[:i+1].sum() for i in range(nprob.size)])
    icdf = interpolate(cdf[:], 0.5*(lmm2[:-1] + lmm2[1:]))
    
    #Sample random points from uniform distribution and find corresponding mass
    ran = np.random.uniform(0, 1, N0)
    hmatch = np.exp(icdf(ran))
    hmatch.sort()
    return hmatch[::-1]
