#Taken from https://github.com/bccp/nbodykit/blob/v0.1.x/contrib/MWhiteHaloFile.py

import numpy


class MWhiteHaloFile(object):
    """
    Halo catalog file using Martin White's format    
    """
    def __init__(self, filename, boxsize = 1380., rsd = False):
        self.filename = filename
        with open(self.filename, 'r') as ff:
            self._nhalo = int(numpy.fromfile(ff, 'i4', 1)[0])
            
            # first entry of each property is zero 
            # for some reason
            self.nhalo = self._nhalo - 1 
    
    def read(self, column):
        """
        Read a data column from the catalogue
        Parameters
        ----------
        column : string
            column to read: CenterOfMass or Mass
        
        Returns
        -------
            the data column; all halos are returned.
        """
        if column == 'Position':
            return self.read_pos()
        elif column == 'Mass':
            return self.read_mass()
        elif column == 'Velocity':
            return self.read_vel()
        else:
            raise KeyError("column `%s' unknown" % str(column))

    def read_mass(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4, 0)
            return numpy.fromfile(ff, count=self._nhalo, dtype='f4')[1:]

    def read_pos(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4 + 5*self._nhalo * 4, 0)
            dtype = numpy.dtype(('f4', 3))
            return numpy.fromfile(ff, count=self._nhalo, dtype=dtype)[1:]

    def read_vel(self):
        with open(self.filename, 'r') as ff:
            ff.seek(4 + 10*self._nhalo * 4, 0)
            dtype = numpy.dtype(('f4', 3))
            return numpy.fromfile(ff, count=self._nhalo, dtype=dtype)[1:]
            
            
def readall(filename, BoxSize = 1380., rsd = False):
    '''Function which instantiates the previous class to read the catalog'''
    dtype = numpy.dtype([
        ('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('LogMass', 'f4'),
        ('Weight', 'f4'),
        ('Mass', 'f4'), ])

    hf = MWhiteHaloFile(filename)
    nhalo = hf.nhalo
    P = numpy.empty(nhalo, dtype)

    P['Position']= numpy.float32(hf.read_pos())
    P['Velocity']= numpy.float32(hf.read_vel())
    P['Mass'] = numpy.float32(hf.read_mass())
    P['LogMass'] = numpy.log10(P['Mass'])

    P['Position'][:] *= BoxSize
    P['Velocity'][:] *= BoxSize

    if rsd:
        dir = "xyz".index(self.rsd)
        P['Position'][:, dir] += P['Velocity'][:, dir]
        P['Position'][:, dir] %= self.BoxSize[dir]

    return {key: P[key] for key in P.dtype.names}
