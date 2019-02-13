#!/usr/bin/env python


import sys
import math
import subprocess
import timeit
import pickle
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from visualization import visualize3d, visualizeMap


class Interpolate:
    def __init__(self, filepath=None):
        if filepath is None:
            filepath = sys.argv[1]
        self.fits = fits.open(filepath)
        print(self.__repr__())

        self.rot_matrix = None
        self.data = None

        self.spacing = None
        self.offset = None

    def __repr__(self):
        repr_str = "****** INTERPOLATE OBJECT ****** \n" + \
            "Number of skewers in the fits file are {}.".format(len(self.fits) - 1)
        return repr_str

    def __len__(self):
        return len(self.fits) - 1

    def get_data(self, sample_rate=1, sample_size=None, noise=0.1, viz=True):
        """

        :param sample_rate:
        :param sample_size:
        :param noise:
        :param viz:
        :return:
        """
        print('****** GETTING DATA ******')

        if sample_size is None:
            sample_size = int(len(self) * sample_rate)

        idx = np.random.choice(np.arange(1, len(self)+1), size=sample_size, replace=False)

        print('Reading {} skewers out of {} skewers.'.format(sample_size, len(self)))

        x_list, y_list, z_list, v_list, s_list = [], [], [], [], []

        for i in idx:
            x, y, z, v = self.__read_skewer__(self.fits[i])
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            v_list.append(v)
            s_list.append([i] * len(x))

        x = np.hstack(x_list)
        y = np.hstack(y_list)
        z = np.hstack(z_list)
        v = np.hstack(v_list)
        s = np.hstack(s_list)

        # rotating the points
        rot_matrix = self.__get_matrix__()
        x, y, z = np.dot(rot_matrix, np.array([x, y, z]))

        # generating noise data
        n = np.ones_like(x) * noise

        # concatenating together
        self.data = np.vstack([x, y, z, n, v, s]).T

        if viz:
            visualize3d(x, y, z)

        print('Read {} points from {} skewers.'.format(len(x), sample_size))

    def __get_matrix__(self):
        """
        Method for calculating rotation matrix
        :return:
        """
        ra, dec = self.__read_plate__()
        mean_ra, mean_dec = np.nanmean(ra), np.nanmean(dec)

        # Vector perpendicular to the above vector over which we rotate the co-ordinate system
        k_perp = np.array([-np.sin(mean_ra), np.cos(mean_ra), 0])

        # Rotation direction - Cross product matrix
        K = np.array([
            [0, -k_perp[2], k_perp[1]],
            [k_perp[2], 0, -k_perp[0]],
            [-k_perp[1], k_perp[0], 0]
        ])

        # ADOPTED FROM: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        # Rotation Matrix to rotate the points vectors
        self.rot_matrix = np.eye(3) - np.sin(mean_dec) * K + (1 - np.cos(mean_dec)) * np.matmul(K, K)

        print('Rotation Matrix: ')
        print(self.rot_matrix)

        return self.rot_matrix


    def __read_plate__(self):
        """
        Method for getting the angular positions and redshift of the skewers
        :return:
        """
        ra_list, dec_list, zq_list = [], [], []

        for i in range(1, len(self.fits)):
            header = self.fits[i].header
            ra_list.append(header['RA'])
            dec_list.append(header['DEC'])
            # zq_list.append(header['Z'])

        self.ra = np.deg2rad(ra_list)
        self.dec = np.deg2rad(dec_list)
        # self.zq = np.array(zq_list)
        return self.ra, self.dec

    @staticmethod
    def __read_skewer__(skewer):
        """
        Method for reading the angular positions, redshift, delta_t value, etc. for a skewer from fits file
        :param skewer: the skewer object in a fits file
        :return: lists of x coordinates, y coordinates, z coordinates, and delta_t values
        """
        ra_deg = skewer.header['RA']
        ra = np.deg2rad(ra_deg)
        dec_deg = skewer.header['DEC']
        dec = np.deg2rad(dec_deg)
        rcomov = skewer.data['RCOMOV']
        value = skewer.data['DELTA_T']

        x = np.sin(dec) * np.cos(ra) * rcomov
        y = np.sin(dec) * np.sin(ra) * rcomov
        z = np.cos(dec) * rcomov
        v = value

        return x, y, z, v

    def process_cube(self, spacing=4, viz=True):

        print('****** PROCESSING WHOLE DATA CUBE ******')

        self.__dachshund__(self.data, spacing, viz=viz)

        print('****** FINISHED PROCESSING ******')

    def process_box(self, spacing=4, xcuts=None, ycuts=None, zcuts=None, viz=True):

        x, y, z = self.data[:, :3].T

        print('****** PROCESSING BOX OF ({} ~ {}, {} ~ {}, {} ~ {}) ******'
              .format(x.min(), x.max(), y.min(), y.max(), z.min(), z.max()))

        # idx = (x >= x.min() + xcuts[0] * (x.max() - x.min())) & \
        #       (x < x.min() + xcuts[1] * (x.max() - x.min())) & \
        #       (y >= y.min() + ycuts[0] * (y.max() - y.min())) & \
        #       (y < y.min() + ycuts[1] * (y.max() - y.min())) & \
        #       (z >= z.min() + zcuts[0] * (z.max() - z.min())) & \
        #       (z < z.min() + zcuts[1] * (z.max() - z.min()))

        if xcuts is None:
            xcuts = [x.min(), x.max()]
        if ycuts is None:
            ycuts = [y.min(), y.max()]
        if zcuts is None:
            zcuts = [z.min(), z.max()]

        idx = (x >= xcuts[0]) & (x < xcuts[1]) &\
              (y >= ycuts[0]) & (y < ycuts[1]) &\
              (z >= zcuts[0]) & (z < zcuts[1])

        self.__dachshund__(self.data[idx], spacing, viz=viz)

        print('****** FINISHED PROCESSING ******')

    def process_points(self, spacing=4, n_points=10000, iterate_whole_cube=False, viz=True):

        print('****** PROCESSING WITH {} points ******'.format(n_points))

        z = self.data[:, 2]
        zmin = math.floor(z.min())
        zmax = math.ceil(z.max())
        start = zmin
        end = zmin
        cnt = 0
        while end <= zmax:
            while end <= zmax:
                idx = ((z >= start) & (z < end))
                if idx.sum() >= n_points:
                    break
                end += 1
            print('Processing z range: {} - {}'.format(start, end))

            self.__dachshund__(self.data[idx], spacing, viz=viz, prefix=iterate_whole_cube*('ds'+str(cnt)+'_'))

            start = end
            cnt += 1
            if not iterate_whole_cube:
                break

        print('****** FINISHED PROCESSING ******')

    @staticmethod
    def __dachshund__(data, spacing, viz, prefix=''):

        print('****** RUNNING DACHSHUND ******')

        # data to send to Dachshund - remove the skewer index column
        chunk = data[:, :-1]

        # skewer index column
        sx = data[:, -1]

        # shift the cordinates to the edge of the cube
        chunk[:, 0] -= chunk[:, 0].min()
        chunk[:, 1] -= chunk[:, 1].min()
        chunk[:, 2] -= chunk[:, 2].min()

        # write binary file for dachshund
        chunk.tofile('pixel_data.bin')

        # write binary file for record
        bin_file = prefix + 'pixel_data.bin'
        if prefix != '':
            chunk.tofile(bin_file)

        # write numpy file
        np_file = prefix + 'pixel_data.npy'
        np.save(np_file, data)

        # calculate values
        x, y, z = chunk[:,:3].T

        # length along each direction - starts at 0
        lx = x.max() - x.min()
        ly = y.max() - y.min()
        lz = z.max() - z.min()

        # Number of parts in which to divide the length along each direction
        npx_x = int(lx // spacing)
        npx_y = int(ly // spacing)
        npx_z = int(lz // spacing)
        print("Grid dimensions are ({},{},{})".format(npx_x, npx_y, npx_z))

        # write config file
        cfg_file = prefix + 'void.cfg'
        with open(cfg_file, 'w') as cf:
            cf.write("lx = %f\n" % lx)
            cf.write("ly = %f\n" % ly)
            cf.write("lz = %f\n" % lz)
            cf.write("num_pixels = %i\n" % len(x))
            cf.write("map_nx = %i\n" % npx_x)
            cf.write("map_ny = %i\n" % npx_y)
            cf.write("map_nz = %i\n" % npx_z)
            cf.write("corr_var_s = 0.23\n")
            cf.write("corr_l_perp = 5\n")
            cf.write("corr_l_para = 5\n")
            cf.write("pcg_tol = 1.0e-5\n")
            cf.write("pcg_max_iter = 1000\n")

        # run dachshund
        message1 = subprocess.run(['./dachshund.exe', cfg_file], stdout=sys.stdout)
        print(message1)

        # rename map file
        map_file = prefix + 'map.bin'
        message2 = subprocess.run(['mv', 'map.bin', map_file], stdout=sys.stdout)
        print(message2)

        # remove additional pixel data binary file
        if prefix != '':
            massage3 = subprocess.run(['rm', 'pixel_data.bin'], stdout=sys.stdout)
            print(massage3)

        if viz:
            visualizeMap(np_file, map_file, cfg_file, cuts={'x': [], 'y': [], 'z': [1, 2, 3]},
                         width=2, prefix=prefix)


if __name__ == '__main__':
    interpolate = Interpolate('delta_transmission_RMplate.fits')
    interpolate.get_data()
    interpolate.process_cube(spacing=10)
    # interpolate.process_box(spacing=4, xcuts=[-30, 30], ycuts=[-30, 30], zcuts=[3600, 3630])
