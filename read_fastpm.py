import os
import numpy

def read_real(filename, size):
    data = numpy.zeros((size, size, size), dtype='f4')
    i = 0
    while True:
        fn = '%s.%03d' % (filename, i)
        geofn = '%s.%03d.geometry' % (filename, i)
        if not os.path.exists(fn):
            if i == 0:
                fn = filename
                geofn = '%s.geometry' % filename
                if not os.path.exists(fn):
                    raise OSError("File not found")
            else:
                break
        d = numpy.fromfile(fn, dtype='f4')
        strides = numpy.loadtxt(open(geofn).readlines()[3].split()[1:], dtype=int)
        offset = numpy.loadtxt(open(geofn).readlines()[1].split()[1:], dtype=int)
        shape = numpy.loadtxt(open(geofn).readlines()[2].split()[1:], dtype=int)    
        ind = tuple([slice(x, x+o) for x, o in zip(offset, shape)])        
        d = numpy.lib.stride_tricks.as_strided(d, shape=shape, strides=strides * 4)
        #print ind, d.shape, d.max()
        data[ind] = d
        i = i + 1
    return data

def read_complex(filename, size):
    data = numpy.zeros((size, size, size // 2 + 1), dtype='complex64')
    i = 0
    while True:
        fn = '%s.%03d' % (filename, i)
        geofn = '%s.%03d.geometry' % (filename, i)
        #print fn
        if not os.path.exists(fn):
            if i == 0:
                fn = filename
                geofn = '%s.geometry' % filename
                if not os.path.exists(fn):
                    raise OSError("File not found")
            else:
                break
        d = numpy.fromfile(fn, dtype='complex64')
        strides = numpy.loadtxt(open(geofn).readlines()[7].split()[1:], dtype=int)
        offset = numpy.loadtxt(open(geofn).readlines()[5].split()[1:], dtype=int)
        shape = numpy.loadtxt(open(geofn).readlines()[6].split()[1:], dtype=int)    
        ind = tuple([slice(x, x+o) for x, o in zip(offset, shape)])        
        d = numpy.lib.stride_tricks.as_strided(d, shape=shape, strides=strides * 8)
        #d = d.view(dtype='complex64')
        #print ind, d.shape, d.max()
        data[ind] = d
        i = i + 1
    return data
 

#size is the size of the array that you want to read in 
#This will determine the kmax by  size*pi/boxsize
#Its assumed that the complex files to be read in are on a 
#finer mesh, hopefully a multiple of this size
#Haven't cheked the function if its not a multiple, 
#its possible it might not work
#Its assumed X is continuous and Z keeps only positive frequencies

def read_complex_sub(filename, size):
    data = numpy.zeros((size, size, size // 2 + 1), dtype='complex64')
    i = 0
    while True:
        fn = '%s.%03d' % (filename, i)
        geofn = '%s.%03d.geometry' % (filename, i)
        #print fn
        if not os.path.exists(fn):
            if i == 0:
                fn = filename
                geofn = '%s.geometry' % filename
                if not os.path.exists(fn):
                    raise OSError("File not found")
            else:
                break
        d = numpy.fromfile(fn, dtype='complex64')
        strides = numpy.loadtxt(open(geofn).readlines()[7].split()[1:], dtype=int)
        offset = numpy.loadtxt(open(geofn).readlines()[5].split()[1:], dtype=int)
        shape = numpy.loadtxt(open(geofn).readlines()[6].split()[1:], dtype=int)    
        ind = tuple([slice(x, x+o) for x, o in zip(offset, shape)])        
        d = numpy.lib.stride_tricks.as_strided(d, shape=shape, strides=strides * 8)

        #subsapmling start, calculate the bounds of 'd' to be saved and corresponding
        #indices of data
        bounds = [[x, x+o] for x, o in zip(offset, shape)]
        save =1 
        sub = size/2
        neglim = shape[0] - sub
        
        #X
        dtupx1 = slice(0, sub, None)
        indx1 =  slice(0, sub, None)
        dtupx2 = slice(neglim, shape[0], None)
        indx2 =  slice(sub, size, None)

        #Y
        if bounds[1][0] >= sub:
            if bounds[1][0] >= neglim:
                dtupy = slice(0, shape[1])
                indy = slice(bounds[1][0] - neglim + sub, bounds[1][1] - neglim + sub)
            elif bounds[1][1] > neglim:
                diff = neglim - bounds[1][0]
                dtupy = slice(diff, bounds[1][1])
                indy = slice(sub, bounds[1][1] - neglim + sub)
            else:
                save = 0
        elif bounds[1][1] > sub:
            diff = sub - bounds[1][0]
            dtupy = slice(0, diff, None)    
            indy = slice(bounds[1][0], sub)
        else:
            dtupy = slice(0, shape[1], None)
            indy = slice(bounds[1][0], bounds[1][1])

        #Z
        if bounds[2][0] >= sub:
            save = 0 
        elif bounds[2][1] > sub:
            diff = sub - bounds[2][0]
            dtupz = slice(0, diff, None)    
            indz = slice(bounds[2][0], sub)
        else:
            dtupz = slice(0, shape[2], None)
            indz = slice(bounds[2][0], bounds[2][1])
            
        if save:
            ind = tuple([indx1, indy, indz])
            dind = tuple([dtupx1, dtupy, dtupz])
            data[ind] = d[dind]
            ind = tuple([indx2, indy, indz])
            dind = tuple([dtupx2, dtupy, dtupz])
            data[ind] = d[dind]
            
        i = i + 1
    return data

