#!/usr/bin/env python


import sys
import math
import subprocess
import numpy as np

from astropy.io import fits


class Interpolate:
    def __init__(self, filepath=None):
        if filepath is None:
            filepath = sys.argv[1]
        self.data = fits.open(filepath)

    def read(self, sample_size=None, sample_rate=None, ra_mean=0, dec_mean=0, noise=0):
        print('******READING******')
        length = len(self.data)
        if sample_size is not None:
            idx = np.random.choice(np.arange(1, length), size=sample_size, replace=False)
        elif sample_rate is not None:
            idx = np.random.choice(np.arange(1, length), size=int((length-1)*sample_rate), replace=False)
        else:
            raise ValueError()

        x_list = []
        y_list = []
        z_list = []
        v_list = []
        for i in idx:
            x, y, z, v = self.read_line(self.data[i], ra_mean=ra_mean, dec_mean=dec_mean)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            v_list.append(v)

        # concatenating together
        x = np.hstack(x_list)
        y = np.hstack(y_list)
        z = np.hstack(z_list)
        v = np.hstack(v_list)

        # shifting cordinates
        x -= x.min() - 5
        y -= y.min() - 5
        z -= z.min() - 5

        # generating noise data
        n = np.ones_like(x) * noise

        # concatenating together
        self.p = np.vstack([x, y, z, n, v]).T

        print('Read {} points from the file'.format(self.p.shape[0]))

    def read_line(self, line, ra_mean=0, dec_mean=0):
        ra = (line.header['RA'] - ra_mean) * np.pi / 360
        dec = (line.header['DEC'] - dec_mean) * np.pi / 360
        rcomov = line.data['RCOMOV']
        value = line.data['DELTA_T']

        x = np.sin(dec) * np.cos(ra) * rcomov
        y = np.sin(dec) * np.sin(ra) * rcomov
        z = np.cos(dec) * rcomov
        v = value

        return x, y, z, v

    def process(self, chunk_size=10000):
        print('******PROCESSING******')
        z = self.p[:,2]
        zmin = math.floor(z.min())
        zmax = math.ceil(z.max())
        start = zmin
        end = zmin
        while end <= zmax:
            while end <= zmax:
                idx = ((z >= start) & (z < end))
                if idx.sum() >= chunk_size:
                    break
                end += 1
            print('Processing z range: {} - {}'.format(start, end))
            self.dachshund(self.p[idx])
            start = end
        print('******FINISHED******')

    def dachshund(self, chunk):

        # write binary file
        chunk.tofile('pixel_data.bin')

        # calculate values
        x = chunk[:,0]
        y = chunk[:,1]
        z = chunk[:,2]
        npx_x = np.floor((x.max() - x.min()) / 4.)
        npx_y = np.floor((y.max() - y.min()) / 4.)
        npx_z = np.floor((z.max() - z.min()) / 4.)

        # write config file
        cf = open("void.cfg", "w")
        cf.write("lx = %f\n" % (x.max() + 5))
        cf.write("ly = %f\n" % (y.max() + 5))
        cf.write("lz = %f\n" % (z.max() + 5))
        cf.write("num_pixels = %i\n" % len(x))
        cf.write("map_nx = %i\n" % npx_x)
        cf.write("map_ny = %i\n" % npx_y)
        cf.write("map_nz = %i\n" % npx_z)
        cf.write("corr_var_s = 0.05\n")
        cf.write("corr_l_perp = 2.5\n")
        cf.write("corr_l_para = 2.5\n")
        cf.write("pcg_tol = 1.0e-5\n")
        cf.write("pcg_max_iter = 1000\n")
        cf.close()

        message = subprocess.run(['./dachshund.exe', 'void.cfg'], stdout=subprocess.PIPE)
        print(message)


if __name__ == '__main__':
    interpolate = Interpolate()
    interpolate.read(sample_rate=1, ra_mean=213.704, dec_mean=53.083, noise=0.1)
    interpolate.process()