#!/usr/bin/env python


import sys
import math
import subprocess
import re
import pickle
import numpy as np
from astropy.io import fits


class Interpolate:
    def __init__(self, filepath=None, sample_size=None, sample_rate=None, noise=0):
        if filepath is None:
            filepath = sys.argv[1]
        self.data = fits.open(filepath)
        self.Q = None
        self.p = None

        self.get_matrix()
        self.get_data(sample_size, sample_rate, noise)

    def get_matrix(self):
        # extract the ra, dec and quasar redshift for each object
        # append a dummy index at the first location since the table data in the fits
        # file starts at index 1
        ra_list, dec_list, z_list = [np.nan], [np.nan], [np.nan]

        for i in range(1, len(self.data)):
            header = self.data[i].header
            ra_list.append(header['RA'])
            dec_list.append(header['DEC'])
            # z_list.append(header['Z'])

        ra, dec = np.deg2rad(np.array(ra_list)), np.deg2rad(np.array(dec_list))
        mean_ra, mean_dec = np.nanmean(ra), np.nanmean(dec)

        # The k-vector pointing along the direction of LOS
        k_par = np.array([np.sin(mean_dec) * np.cos(mean_ra), np.sin(mean_dec) * np.sin(mean_ra), np.cos(mean_dec)])

        # Vector perpendicular to the above vector over which we rotate the co-ordinate system
        k_perp = np.array([-np.sin(mean_ra), np.cos(mean_ra), 0])

        # Rotation direction - Cross product matrix
        K = np.array([
            [0, -k_perp[2], k_perp[1]],
            [k_perp[2], 0, -k_perp[0]],
            [-k_perp[1], k_perp[0], 0]
        ])

        # ADOPTED FROM: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        # Rotation Matrix to rotate the basis vectors
        R = np.eye(3) + np.sin(mean_dec) * K + (1 - np.cos(mean_dec)) * np.matmul(K, K)

        # Rotation Matrix to rotate the points vectors
        Q = np.eye(3) - np.sin(mean_dec) * K + (1 - np.cos(mean_dec)) * np.matmul(K, K)

        self.Q = Q

    def get_data(self, sample_size, sample_rate, noise, viz=False):
        length = len(self.data)
        if sample_size is not None:
            idx = np.random.choice(np.arange(1, length), size=sample_size, replace=False)
        elif sample_rate is not None:
            idx = np.random.choice(np.arange(1, length), size=int((length-1)*sample_rate), replace=False)
        else:
            raise ValueError()

        x_list, y_list, z_list, v_list = [], [], [], []
        for i in idx:
            x, y, z, v = self.read_line(self.data[i])
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            v_list.append(v)

        # concatenating together
        x = np.hstack(x_list)
        y = np.hstack(y_list)
        z = np.hstack(z_list)
        v = np.hstack(v_list)

        # rotating the points
        x, y, z = np.dot(self.Q, np.array([x, y, z]))

        # shifting cordinates
        x -= x.min() - 5
        y -= y.min() - 5
        z -= z.min() - 5

        # generating noise data
        n = np.ones_like(x) * noise

        # concatenating together
        self.p = np.vstack([x, y, z, n, v]).T

        if viz:
            self.visualize3d(x, y, z)

    def read_line(self, line):
        ra = (line.header['RA']) * np.pi / 180
        dec = (line.header['DEC']) * np.pi / 180
        rcomov = line.data['RCOMOV']
        value = line.data['DELTA_T']

        x = np.sin(dec) * np.cos(ra) * rcomov
        y = np.sin(dec) * np.sin(ra) * rcomov
        z = np.cos(dec) * rcomov
        v = value

        return x, y, z, v

    def timeit(self, chunk_size=None, sample_rate=None):
        if chunk_size is not None:
            z = self.p[:,2]
            zmin = math.floor(z.min())
            zmax = math.ceil(z.max())
            start = zmin
            end = zmin
            while end <= zmax:
                idx = ((z >= start) & (z < end))
                if idx.sum() >= chunk_size:
                    break
                end += 1
        elif sample_rate is not None:
            z = self.p[(self.p[:,2] < 300) & (self.p[:,2] > 200)]
            n = z.shape[0]
            idx = np.random.choice(n, int(n*sample_rate))
        output = self.dachshund(self.p[idx], 0)
        time = self.extract_time(output)
        return time

    def dachshund(self, chunk, i):

        # write binary file
        chunk.tofile('pixel_data.bin')

        # calculate values
        x = chunk[:,0]
        y = chunk[:,1]
        z = chunk[:,2]

        npx_x = int(np.floor((x.max() - x.min()) / 2.))
        npx_y = int(np.floor((y.max() - y.min()) / 2.))
        npx_z = int(np.floor((z.max() - z.min()) / 2.))

        # write config file
        cf = open("void.cfg", "w")
        cf.write("lx = %f\n" % (x.max() + 5))
        cf.write("ly = %f\n" % (y.max() + 5))
        cf.write("lz = %f\n" % (z.max() + 5))
        cf.write("num_pixels = %i\n" % len(x))
        cf.write("map_nx = %i\n" % npx_x)
        cf.write("map_ny = %i\n" % npx_y)
        cf.write("map_nz = %i\n" % npx_z)
        cf.write("corr_var_s = 0.23\n")
        cf.write("corr_l_perp = 5\n")
        cf.write("corr_l_para = 5\n")
        cf.write("pcg_tol = 1.0e-5\n")
        cf.write("pcg_max_iter = 1000\n")
        cf.close()

        message = subprocess.run(['./dachshund.exe', 'void.cfg'], stdout=subprocess.PIPE).stdout.decode("utf-8")
        return message

    @staticmethod
    def extract_time(output):
        for line in output.splitlines():
            if 'Total time' in line:
                time = re.search(r'\d+\.?\d*', line)
                return float(time.group(0))


def time_size(sizes):
    times = []
    for size in sizes:
        print('Testing chunk size = {}'.format(size))
        interpolate = Interpolate('delta_transmission_RMplate.fits', sample_rate=1, noise=0.1)
        tentime = 0
        for i in range(10):
            time = interpolate.timeit(chunk_size=size)
            print('Interpolation {} took: {} seconds'.format(i, time))
            tentime += time
        times.append(tentime / 10)
    return times


def time_rate(rates):
    times = []
    for rate in rates:
        print('Testing sample rate = {}'.format(rate))
        interpolate = Interpolate('delta_transmission_RMplate.fits', sample_rate=1, noise=0.1)
        tentime = 0
        for i in range(10):
            time = interpolate.timeit(sample_rate=rate)
            print('Interpolation {} took: {} seconds'.format(i, time))
            tentime += time
        times.append(tentime / 10)
    return times


if __name__ == '__main__':
    rates = [0.4, 0.5]
    times = time_rate(rates)
