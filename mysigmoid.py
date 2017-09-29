"""                                                                                                               This module
"""
import numpy, math
from numpy import log10 as log10
from numpy import pi as pi
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.optimize import curve_fit as cf
from scipy.optimize import minimize
from scipy.integrate import simps, quad
from pmesh.particlemesh import ParticleMesh
import sys
sys.path.append("/global/homes/c/chmodi/Programs/Py_codes/modules/")
import mytools as tools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from sklearn import svm

##slope = 0
intercept = 0
#########################################################################
def sm_scale(mass, ovd = 178., rhoc = 27.756, M = 0.3175):
    '''Return Eulerian scale of the halo for solar mass'''    
    rhobar = ovd * rhoc*M*10**10.
    return (3 * mass / (4*numpy.pi*rhobar))**(1/3.)    


def sigmoidf(delta, hold, alpha = 0.8, b = 1):
    '''Pass delta through a sigmoid of threshold = hold, multiplicative coeff = alpha, power = b'''
    z = alpha * (delta**b - hold**b)    
    return 1./(1 + numpy.exp(-z))

def normalize(ar):
    '''Normalize the array by subtracting the mean and dividing by std
    '''
    mean = ar.mean()
    std = ar.std()
    return (ar - mean)/std

def svm_thresh(ft, cls, cls_wt = None):
    '''Do the svm on the ft(feature) and cl(ass) 
    '''
    if cls_wt is None:
        clf = svm.LinearSVC(max_iter=10000, dual = False)
    else:
        clf = svm.LinearSVC(max_iter=10000, dual = False, class_weight = cls_wt)
    clf.fit(ft, cls)
    return clf.coef_[0], clf.intercept_[0], clf.score(ft, cls)


def find_mask(ftlist, cls,  mesh, catpos,  alpha = 20., b = 1, file = None, label = "None", cls_wt = None, NN = False):
    '''Find mask passed through sigmoid'''
    mesh.push()
    nft = len(ftlist)
    ft = numpy.empty([cls.size, nft])
    
    if NN:
        nc = mesh.real.shape[0]
        bs = mesh.BoxSize[0]
        side = bs/nc
        pos = numpy.round(catpos/side).astype(int)
        pos[pos == nc] = 0
        for foo in range(nft):
            ft[:, foo] = ftlist[foo][tuple(pos.T)]
        
    else:
        for foo in range(nft):
            mesh.clear()
            mesh.real[:] = ftlist[foo]
            ft[:, foo] = mesh.readout(catpos)
    
    coeff, intercept, score = svm_thresh(ft, cls, cls_wt)
    if file is not None:
        file.write("label = %s , nft = %d\n"%(label, nft))
        file.write("coeff = %s, intercept = %s \n"%(str(coeff), str(intercept)))
        file.write("threshold = %s, score = %0.3f \n"%(str(intercept/coeff[0]), score))

    pmgrid = numpy.zeros_like(mesh.real)
    for foo in range(nft):
        pmgrid += coeff[foo]*ftlist[foo]
        
    mask = sigmoidf(pmgrid, -intercept, alpha, b)

    mesh.pop()
    return mask, coeff, intercept
    

def crosspower(mesh1, mesh2, bs, a1 = False, a2 = False, cross = True, normit = False, std1 = False, unlog  = False, ovd1 = False, ovd2 = False):
    pmesh1 = mesh1.copy()
    pmesh2 = mesh2.copy()
    if unlog:
        pmesh1 =10**pmesh1
        pmesh2 =10**pmesh2
    if normit:
        pmesh1 = normalize(pmesh1)
        pmesh2 = normalize(pmesh2)
    if pmesh1.mean() < 10**-6:
        pmesh1 += 1
    if pmesh2.mean() < 10**-6:
        pmesh2 += 1
    if std1:
        pmesh1 = pmesh1/pmesh1.std()
        pmesh2 = pmesh2/pmesh2.std()
    add1 = 0
    add2 = 0

    if a1:
        k, pow1 = tools.power_cheap(pmesh1 + add1, boxsize = bs)
    if a2:
        k, pow2 = tools.power_cheap(pmesh2 + add2, boxsize = bs)
    if cross:
        k, powc = tools.power_cheap(pmesh1 + add1, pmesh2 + add2, boxsize = bs)

    toret = [k]
    if a1:
        toret.append(pow1)
    if a2:
        toret.append(pow2)
    if cross:
        toret.append(powc)
    return toret

#########################################################################
###Different style of painting and other array transformations

def grid(mesh, pos = None,  wts = None, style = 'CIC', ingrid = None, logscale = False,
         dolog = False, smooth = None, R=0, dologsm = False):
    
    mesh.push()
    if pos is None:
        if ingrid is None:
            return 'No input given! Give either position or a grid to operate on.'
        else:
            mesh.clear()
            mesh.real[:] = ingrid[:]
            if logscale:
                mesh.real[:] = 10**mesh.real[:]
                mesh.real[mesh.real == 1] = 0
    else:
        if wts is None:
            wts = numpy.ones(pos.shape[0])

        mesh.clear()
        if style == 'CIC':
            mesh.paint(pos, mass = wts)
        elif style == 'NN':
            nc = mesh.real.shape[0]
            mesh.real[:], dummy = numpy.histogramdd(pos, bins = (nc, nc, nc), weights=wts)

    toret = mesh.real.copy()

    if dolog:
        dummy = numpy.empty_like(toret)
        dummy[:] = toret[:]
        dummy[dummy <= 0] = 1
        dummy = log10(dummy)
        toret2 = dummy.copy()

    if smooth is not None:
        mesh.clear()
        mesh.real[:] = toret[:]
        mesh.r2c()
        if smooth == 'Fingauss':
            mesh.transfer([tools.pmFingauss(R)])
        elif smooth == 'Gauss':
            mesh.transfer([tools.pmGauss(R)])
        elif smooth == 'Tophat':
            mesh.transfer([tools.pmTophat(R)])
        else:
            return 'Smoothing kernel not defined'
        mesh.c2r()
        toret3 = mesh.real.copy()

        if dologsm:
            dummy = numpy.empty_like(toret3)
            dummy[:] = toret3[:]
            dummy[dummy <= 0] = 1
            dummy = log10(dummy)
            toret4 = dummy.copy()
    
    mesh.pop()
    if dolog:
        if smooth:
            if dologsm:
                return toret, toret2, toret3, toret4
            else:
                return toret, toret2, toret3
        else:
            return toret, toret2
    elif smooth:
        if dologsm:
            return toret, toret3, toret4
        else:
            return toret, toret3
    else:
        return toret

    

#########################################################################
#### Stellar Mass

def line(x, m , c):
    return m*x + c

def scatter(hmass, sigma, seed = 123):
    '''Take in halo mass in solar mass and return the log-normally scattered mass
    '''
    logl = numpy.log10(hmass)
    rng = numpy.random.RandomState(seed)
    t = rng.normal(scale=sigma, size=len(logl))
    logl = logl + t
    return 10**logl

def stellar_relation():
    x1 = 10**15 * 1.6
    x2 = 10**12 * 4

    y1 = x1 * 1.5*10**-3
    y2 = x2 * 3.2*10**-2
    line1, dummy = cf(line, numpy.array([log10(x1), log10(x2)]), numpy.array([log10(y1), log10(y2)]))

    y1 = x1 * 3.5*10**-4
    y2 = x2 * 2.2*10**-2
    line2, dummy = cf(line, numpy.array([log10(x1), log10(x2)]), numpy.array([log10(y1), log10(y2)]))

    #mean line
    m = (line1[0] + line2[0])*0.5
    c = (line1[1] + line2[1])*0.5

    xval = numpy.linspace(log10(x2), log10(x1))
    xline = line(xval, m, c)
    stellar_sigma = interpolate(xline, 0.01 + (line(xval, line1[0], line1[1]) - xline))
    return stellar_sigma, m, c


def htostar(hmass, seed = 123, stellar_sigma = stellar_relation()[0], slope = stellar_relation()[1],\
                intercept =  stellar_relation()[2]):
    '''Take in halo mass in solar mass and return the stellar mass in solar mass.
    '''
    logh = numpy.log10(hmass)
    rng = numpy.random.RandomState(seed)

    logs = logh*slope + intercept
    testf = lambda x : rng.normal(0, x, 1)
    logs = logs + numpy.array(list(map(testf, stellar_sigma(logs)))).T
    return 10**logs[0, :] 

def twiddle(halomass, sigma, seed=12345):
    length = halomass.copy()
    logl = numpy.log10(length)
    rng = numpy.random.RandomState(seed)
    t = rng.normal(scale=sigma, size=len(logl))
    logl = logl + t
#     logl[0] = np.inf
    arg = logl.argsort()[::-1]
    halos = halomass[arg].copy()
    return halos, arg

#########################################################################
###Functions for fitting mass
def fit_log(x, *p):
   ''' y = b*numpy.log10(1 + a*x) + c'''
   a, b, c = p
   x2 = numpy.log10(1 + a*x)
   return b*x2 + c

def quad_exp(x, *p):
    '''y = 10**z *(ax**2 + b*x + c)'''
    a, b, c, y  = p
    return (10**y) * (a*x**2 + b*x+ c )

def chi_sqf(p, xdata, ydata, sigma):
    model =  quad_exp(xdata, *p)
    return (((model - ydata)/sigma)**2).sum()

def quad_exp_der(p, xdata, ydata, sigma):
    model = quad_exp(xdata, *p)
    fac = (2*(model - ydata)/sigma)
    a, b, c, y  = p
    #a, b, c, y  = p; model = (10**y) * (a*x**2 + b*x+ c )
    dmdy = numpy.log(10)*model*fac
    dmda = (10**y)*(xdata**2)*fac
    dmdb = (10**y)*xdata*fac
    dmdc = (10**y)*fac
    
    return numpy.array([dmda.sum(), dmdb.sum(), dmdc.sum(), dmdy.sum()]).T

def domass_10min(Y, X, func = chi_sqf, p0 = [1, 1, 1, 1], lim = False, sigma = True, absig = False, \
    abund = False, retdata = False, nonzeroy = True, ranzero = 0, tol = 0):
    
    xdata = X.astype("float64").flatten()
    ydata = Y.astype("float64").flatten()
    if nonzeroy:
        pos = numpy.where(ydata > 0)[0]
        ydata = ydata[pos]
        xdata = xdata[pos]
    if abund:
        xdata = numpy.sort(xdata)[::-1]
        ydata = numpy.sort(ydata)[::-1]

    if lim:
        pos = numpy.where(ydata > (lim))[0]
        ydata = ydata[pos]
        xdata = xdata[pos]

    if ranzero:
        posz = numpy.where(ydata == 0)[0]
        posz = numpy.random.permutation(posz)
        pos = numpy.concatenate((pos, posz[:int(ranzero*Y.size/100)]))
        
    if sigma:
        sigmaval = ydata.copy()
        sigmaval[sigmaval == 0] = 1
    else:
        sigmaval = numpy.ones_like(ydata)

#    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
#                   method='L-BFGS-B', options = {'ftol' : 10**-15, 'gtol':10**-10})
    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
                   method='BFGS', options = {'gtol':10**-10})

#     if tol:
#         res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', tol=10**-10)
#     else:res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', options ={'fatol':0.0001})
    print(res.message, res.nit)
    sigmass = quad_exp(X, *res.x)
    if retdata:
        return sigmass, res.x, xdata, ydata
    else:
        return sigmass, res.x


def domass(Y, X, func = fit_log, p0 = [1, 2, 8], lim = False, loglim = False, sigma = True, absig = False, \
    abund = False, retdata = False, nonzeroy = True, ranzero = 0):
    '''Take in X(delta) & Y(loghalo) field and fit with function 'func' and starting parameter 'p0'
    '''
    xdata = X.flatten()
    ydata = Y.flatten()
    if nonzeroy:
        pos = numpy.where(ydata > 0)[0]
        if ranzero:
            posz = numpy.where(ydata == 0)[0]
            posz = numpy.random.permutation(posz)
            pos = numpy.concatenate((pos, posz[:int(ranzero*Y.size/100)]))
        ydata = ydata[pos]
        xdata = xdata[pos]
    if abund:
        #xdata = xdata[xdata > 0].flatten()
        #ydata = ydata[ydata > 0].flatten()
        xdata = numpy.sort(xdata)[::-1]
        ydata = numpy.sort(ydata)[::-1]

    if lim:
        pos = numpy.where(ydata > (lim))[0]
        ydata = ydata[pos]
        xdata = xdata[pos]
    if loglim:
        pos = numpy.where(ydata > log10(lim))[0]
        ydata = ydata[pos]
        xdata = xdata[pos]

    if sigma:
        sigmaval = ydata.copy()
        sigmaval[sigmaval == 0] = 1
        poptall, dummy = cf(func, xdata, ydata, p0 = p0, sigma= 1/sigmaval, \
                            absolute_sigma= absig)
    else:
        poptall, dummy = cf(func, xdata, ydata, p0 = p0)

    sigmass = func(X, *poptall)
    if retdata:
        return sigmass, poptall, xdata, ydata
    else:
        return sigmass, poptall


#########################################################################
def showfield(f1, f2,  f3 = None, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, v1 = None, v2 = None, norm = False, lognorm =False, cmap = 'RdBu_r', t1='Sigmoid', t2='Halo', t3='Difference'):
    '''Show fields f1, f2, f3(f1-f2) in x, y, z range in value range v1-v2, for a normed field on
    a lognorm cmap. Titles are t1, t2, t3
    '''
    if v1 is None:
        v1 = f1.min()
        if f2.min() <v1:
            v1 = f2.min()
    if v2 is None:
        v2 = f1.max()
        if f2.max() > v2:
            v2 = f2.max()
    if x2 is None:
        x2 = f1.shape[0]
    if y2 is None:
        y2 = f1.shape[1]
    if z2 is None:
        z2 = f1.shape[2]
        
    fig, axar = plt.subplots(1, 3, figsize = (18, 6))
    plt.rcParams['image.cmap'] = cmap
    ax = axar[0]
    if norm:
        ax.imshow(normalize(f1)[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)
    else:
        if lognorm:
            ax.imshow(f1[x1:x2, y1:y2, z1:z2].sum(axis = -1), \
                      norm=SymLogNorm(linthresh= 0.0001, linscale=0.0001, vmin =v1, vmax =v2 ))
        else:
            ax.imshow(f1[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)

    ax.set_title("Sigmoid")

    ax = axar[1]
    if norm:
        ax.imshow(normalize(f2)[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)
    else:
        if lognorm:
            ax.imshow(f2[x1:x2, y1:y2, z1:z2].sum(axis = -1), \
                      norm=SymLogNorm(linthresh= 0.0001, linscale=0.0001, vmin =v1,vmax= v2 ))
        else:
            ax.imshow(f2[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)

    ax.set_title("Halos")
    
    ax = axar[2]
    if f3 is None:
        f3 = f1 - f2
    if norm:
        im = ax.imshow(normalize(f3)[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)
    else:
        if lognorm:
            im= ax.imshow(f3[x1:x2, y1:y2, z1:z2].sum(axis = -1), \
                          norm=SymLogNorm(linthresh= 0.0001, linscale=0.0001, vmin=v1, vmax=v2 ))
        else:
            im= ax.imshow(f3[x1:x2, y1:y2, z1:z2].sum(axis = -1), vmin = v1, vmax =v2)
    ax.set_title("Difference")
    cbar_ax = fig.add_axes([0.95, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    return fig, ax



def hist_slice(hgrid, sgrid, mthresh = None, bins = 20, sub1 = False, 
               logscale = False, ran = [-10, 10], zeropoints = False, takelog = False):
    '''Take in two grids, X(hgrid) and Y(sgrid), and calculate histogram on X-Y corresponding to slices in X
    '''
    hflat = hgrid.flatten()
    sflat = sgrid.flatten()
    if takelog:
        dummy = hflat.copy()
        dummy[dummy <= 0 ] = 1
        hflat = log10(dummy)
        dummy = sflat.copy()
        dummy[dummy <= 0 ] = 1
        sflat = log10(dummy)

    if sub1 is True:
        sflat[sflat == 1] = 0
    args = numpy.argsort(hflat)
    if mthresh is None:
        mthresh = [1*10**14, 1*10**13, 1*10**12, 1*10**11, 1*10**10, 1*10**9]
    mthresh = numpy.array(mthresh)
    if logscale:
        mthresh = numpy.log10(mthresh)
    rthresh = [len(hflat)]
    for m in mthresh:
        try:
            rthresh.append(numpy.where(hflat[args] > m)[0][0])
        except:
            rthresh.append(hflat.size)
    hists = []
    edges = []
    for foo in range(len(rthresh) -1):
        dummy, dummy2  = numpy.histogram((hflat - sflat)[args][rthresh[foo + 1]:rthresh[foo]], bins = bins, range = ran)
        hists.append(dummy)
        edges.append((dummy2[1:] + dummy2[:-1])/2.)

    if zeropoints:
        #0 halo
        dummy, dummy2  = numpy.histogram(sflat[hflat == 0], bins = bins, range = ran)
        hists.append(dummy)
        edges.append((dummy2[1:] + dummy2[:-1])/2.)

        #0 sigmoid
        dummy, dummy2  = numpy.histogram(hflat[sflat == 0], bins = bins, range = ran)
        hists.append(dummy)
    
    return hists, edges



def gridscatter(gridP, gridS, mbin):
    '''Take in two grids, X(hgrid) and Y(sgrid), and return the value for gridS in slices of gridP
    '''
    scatter = []
    mhigh = 1e17
    pos = numpy.where(gridP > mbin[0])
    scatter.append([gridP[pos], gridS[pos]])
    for foo in range(len(mbin) - 1):
        mhigh = mbin[foo]
        mlow = mbin[foo+1]
        pos = numpy.where((gridP > mlow) & (gridP < mhigh))
        scatter.append([gridP[pos], gridS[pos]])
        #dummy, dummy2 = numpy.histogram(gridP[pos] - gridS[pos], range = (0.1, 10), bins = 100)
    return scatter    


def plot_pdf(histlist, edges, shape = None, size = None, labels = None, xlim = None,\
                titles = None, sharex= False, sharey = False, logscale = False,\
             dofit = False, cscheme = ['r', 'b', 'g'], fig = None, axar = None, fontsize = 12):
    '''Create different hitograms (pdf) on a (n//3)*3 grid on linear/log scale
    '''
    if shape is None:
        shape = [len(edges) //3, 3]
    if size is None:
        size =  [3*4, (len(edges)//3)*3]
    if fig is None:
        fig, axar = plt.subplots(shape[0], shape[1], \
                             figsize = (size[0],size[1]), sharex=sharex, sharey=sharey)

    nhist = len(histlist)
    if labels is None:
        labels = [None]*nhist
    
    coeff = [0]
    for foo in range(shape[0]):
        for boo in range(shape[1]):
            ax = axar[foo, boo]          
            if xlim is None:
                ax.set_xlim(-5, 10)
            else:
                ax.set_xlim(xlim[0], xlim[1])
            if logscale:
                ax.set_yscale('log')
            shoo = shape[1]*foo + boo
            if titles is not None:
                ax.set_title(titles[shoo], fontsize = fontsize)
            x = edges[shoo].copy()
            width = (x[1] - x[0])/float(nhist)
            
            for n in range(nhist):
                y = histlist[n][shoo].copy()
                y2 = numpy.zeros_like(y)
                for i in range(y.size):
                    y2[i] = y[:i].sum()
                if dofit:
                    try:
                        coeff, var_matrix = cf(fit_gauss, x, y, p0=[2*y.max(), 0., 0.1])
                        yc = fit_gauss(x, *coeff)
                        ax.plot(x, yc, color = cscheme[n], alpha = 0.5,
                        label =  "$G\ x_0 = %0.3f$ \n $\sigma = %0.3f$"%(ceoff[1],coeff[2]))
                    except:
                        try:
                            coeff, var_matrix = cf(fit_lorentz, x, y, p0=[2*y.max(), 0., 0.1])
                            yc = fit_lorentz(x, *coeff)
                            ax.plot(x, yc, color = cscheme[n], alpha = 0.5,
                                    label =  "$L\ x_0 = %0.3f$ \n $\gamma = %0.3f$"%(coeff[1],coeff[2]))
                        except:
                            print('No converge')
                            coeff = [0]
                if (foo + boo) == 0:
                    try:
                        ax.bar(x+ width*n, y, align='center', width= width, \
                           color= cscheme[n], label = labels[n], alpha = 0.5)
                    except:
                        pass
                else:
                    try:
                        ax.bar(x+ width*n, y, align='center', width= width, \
                           color= cscheme[n], alpha = 0.5)
                    except:
                        pass
                try:
                    ax.legend(loc = 0, fontsize = fontsize)
                except:
                    pass
    
    plt.tight_layout()
    return fig, axar

def plot_pdfcdf(histlist, edges, shape = None, size = None, labels = None, \
                titles = None, sharex= False,
                sharey = False, dofit = False, cscheme = ['r', 'b', 'g']):
    '''Create a 6*3 plot of the pdf on linear and log scale as well as that of the cdf
    '''
    if shape is None:
        shape = [len(edges), 3]
    if size is None:
        size =  [3*4, len(edges)*3]
    fig, axar = plt.subplots(shape[0], shape[1], \
                             figsize = (size[0],size[1]), sharex=sharex, sharey=sharey)

    nhist = len(histlist)
    if labels is None:
        labels = [None]*nhist
    
    coeff = [0]
    for shoo in range(nhist):
        for foo in range(shape[0]):
            x = edges[shoo].copy()
            width = (x[1] - x[0])/float(nhist)
            y = histlist[shoo][foo].copy()
            y2 = numpy.zeros_like(y)
            for i in range(y.size):
                y2[i] = y[:i].sum()

            if dofit:
                try:
                    coeff, var_matrix = cf(fit_lorentz, x, y, p0=[5000, 0.5, 0.1])
                    yc = fit_lorentz(x, *coeff)
                except:
                    print('No converge')
                    coeff = [0]
                    
            for boo in range(shape[1]):
                ax = axar[foo, boo]
                if titles is not None:
                    ax.set_title(titles[foo])
                ax.set_xlim(-5, 10)
                if boo is 1:
                    ax.set_yscale('log')

                if coeff[0]:
                    ax.plot(x, yc, color = cscheme[shoo], alpha = 0.5,
                    label = labels[shoo] + \
                    "$\ x_0 = %0.3f$ \n $\gamma = %0.3f$"%(coeff[1],coeff[2]))
                        
                if boo < 2:
                    ax.bar(x+ width*shoo, y, align='center', width= width, \
                           color= cscheme[shoo], \
                       label = labels[shoo], alpha = 0.5)
                else:
                    ax.bar(x + width*shoo, y2/y2[-1], align='center', width= width, \
                           label = labels[shoo], color= cscheme[shoo], alpha = 0.5)                    
                    ax.axhline(0.95, color='k')
                    ax.axvline(1, color='k')
                    ax.axvline(-1, color='k')
    if labels[0] is not None:
        for foo in range(shape[0]):
            axar[foo, 0].legend(loc = 0)
    plt.tight_layout()

    return fig, axar



def multi_plot(axar, k, halop, fieldp, crossp,  color, label = "", ls = "-", lw = 1, alpha = 1):
    
    cmap = plt.cm.jet
    cstyle = numpy.empty([cmap.N, 4])
    for i in range(cmap.N):
        cstyle[i,:] = cmap(i)
    cstyle = cstyle[::50, :]
 
    if not type(color) == str:
        color = cstyle[color]
    axar[0].plot(k, crossp/(halop*fieldp)**0.5, label = label, ls = ls, lw = lw, alpha = alpha, color = color)  
    axar[1].plot(k, halop - crossp**2./fieldp, label = label, ls = ls, lw = lw, alpha = alpha, color = color)      
    axar[2].plot(k, fieldp/halop, label = label, ls = ls, lw = lw, alpha = alpha, color =  color)  

    axar[0].set_ylabel("Cross correlation", fontsize = 14)
    axar[0].set_ylim(0.8,1.1)
    axar[0].axhline(1, lw = 0.5)
    axar[0].legend(loc = 0, fontsize = 13, ncol = 2)
    
    axar[1].set_ylim(10**-1, 10**5.)
    axar[1].set_yscale("log")
    axar[1].set_ylabel("Stochasticity", fontsize = 14)
    
    axar[2].set_ylabel("Transfer Function", fontsize = 14)
    axar[2].set_ylim(0.05,2)
    axar[2].axhline(1, lw = 0.5)
    
    for ax in axar:
        ax.set_xscale("log")
        ax.set_xlabel("k (h/Mpc)", fontsize = 14)

############################################################################
#### Fitting functions
def fit_lorentz(x, *p):
    '''A, x0, g; A*g/ ((x - x0)**2 + g**2)
    '''
    A, x0, g = p
    return A*g/ ((x - x0)**2 + g**2)

def fit_gauss(x, *p):
    '''A, mu, sigma; A*numpy.exp(-(x-mu)**2/(2.*sigma**2))/sigma
    '''
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))/sigma

def fit_lognorm(x, *p):
    '''A, mu, sigma; A*numpy.exp(-(numpy.log(x)-mu)**2/(2.*sigma**2))/sigma/x
    '''
    A, mu, sigma = p
    return A*numpy.exp(-(numpy.log(x)-mu)**2/(2.*sigma**2))/sigma/x

def fit_exp(x, *p):
    '''A, x0, l; A*numpy.exp(-l*abs(x - x0))
    '''
    A, x0, l = p
    return A*numpy.exp(-l*abs(x - x0))

from scipy.special import gamma

def fit_student_t(x, *p):
    '''A, x0, nu; A*g/ ((x - x0)**2 + g**2)
    '''
    A, x0, nu = p
    return A*gamma((nu + 1)/2.)*(1 + (x-x0)**2. / nu)**((-nu + 1)/2.)/(numpy.sqrt(nu)*gamma(nu/2.))
