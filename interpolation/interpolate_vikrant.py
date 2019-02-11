#!/usr/bin/env python
import sys
import subprocess
import numpy as np
import helper

from data_wrangle import celestial_rot_matrix
from astropy.io import fits


class Interpolate:
    def __init__(self, name, filepath=None, spacing=4.):
        if filepath is None:
            filepath = sys.argv[1]
        self.data_file = fits.open(filepath)

        # will be prefixed to names of all files written
        self.name = name

        self.rot_matrix = None
        self.pixel_data = None

        self.ra = None
        self.dec = None
        self.zq = None

        self.spacing = spacing

        # get locations on the celestial sphere
        self.get_locations()
        self.get_matrix()

    def __repr__(self):
        repr_str = "***** Interpolate Object ***** \n" + \
            "Number of skewers are {}".format(len(self.data_file) - 1)
        return repr_str

    def __len__(self):
        return len(self.data_file) - 1

    def get_locations(self):
        """ Get the angular positions and the redshift
        of the skewers
        """
        ra_list, dec_list, zq_list = [], [], []
        for i in range(1, len(self.data_file)):
            hdr = self.data_file[i].header

            ra_list.append(hdr['RA'])
            dec_list.append(hdr['DEC'])
            zq_list.append(hdr['Z'])

        self.ra = np.deg2rad(ra_list)
        self.dec = np.deg2rad(dec_list)
        self.zq = zq_list

    def get_matrix(self, center=None):
        """ Rotate the celestial sphere such that the polar axis
        points towards the central ra and dec of the data patch
        """
        print("****** CALCULATING ROTATION MATRIX ********")
        if center is None:
            center = [np.nanmean(self.ra), np.nanmean(self.dec)]

        self.rot_matrix = celestial_rot_matrix(*center, is_rad=True)

    def get_data(self, skewers_num=None, skewers_perc=1., noise=0.01):
        """ Load skewer data and convert to cartesian cordinates
        Parameters:
        ===========
        skewers_num : select a given number of skewers
        skewers_perc : select a given percentage of skewers
        noise : measurement noise
        """
        if skewers_num is None:
            skewers_num = int((len(self) - 1) * skewers_perc)
        ixs = np.random.choice(np.arange(1, len(self)), size=skewers_num,
                               replace=False)

        x_list, y_list, z_list, v_list = [], [], [], []
        s_list = []

        print('****** READING ******')
        for i in ixs:
            rcomov = self.data_file[i].data['RCOMOV']
            value = self.data_file[i].data['DELTA_T']

            x = np.sin(self.dec[i-1]) * np.cos(self.ra[i-1]) * rcomov
            y = np.sin(self.dec[i-1]) * np.sin(self.ra[i-1]) * rcomov
            z = np.cos(self.dec[i-1]) * rcomov

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            v_list.append(value)
            s_list.append([i] * len(x))

        # concatenating together
        x = np.hstack(x_list)
        y = np.hstack(y_list)
        z = np.hstack(z_list)
        v = np.hstack(v_list)
        sx = np.hstack(s_list)

        # rotating the points
        x, y, z = np.dot(self.rot_matrix, np.array([x, y, z]))

        # generating noise data
        n = np.ones_like(x) * noise

        # final pixel data
        self.pixel_data = np.vstack([x, y, z, n, v, sx]).T

        print('Read {} points from the file'.format(len(x)))

    def process_box(self, xcuts=None, ycuts=None, zcuts=None):
        xx, yy, zz = self.pixel_data[:, :3].T

        if xcuts is None:
            xcuts = [xx.min(), xx.max()]
        if ycuts is None:
            ycuts = [yy.min(), yy.max()]
        if zcuts is None:
            zcuts = [zz.min(), zz.max()]

        ixs = (xx >= xcuts[0]) & (xx < xcuts[1]) &\
              (yy >= ycuts[0]) & (yy < ycuts[1]) &\
              (zz >= zcuts[0]) & (zz < zcuts[1])

        # data to send to Dachsund - remove the skewer index column
        self.chunk = self.pixel_data[:, :-1][ixs]

        # shift the cordinates to the edge of the cube
        self.chunk[:, 0] -= self.chunk[:, 0].min()
        self.chunk[:, 1] -= self.chunk[:, 1].min()
        self.chunk[:, 2] -= self.chunk[:, 2].min()

        self.sx = self.pixel_data[:, -1][ixs]

        self.dachshund()

    def dachshund(self, counter=0):
        # write binary file
        self.chunk.tofile('pixel_data.bin')

        self.pixel_file = self.name + '_pixel_data.bin'
        self.chunk.tofile(self.pixel_file)

        # calculate values
        xx, yy, zz = self.chunk[:, :3].T

        # length along each direction - starts at 0
        self.lx = xx.max()
        self.ly = yy.max()
        self.lz = zz.max()

        # Number of parts in which to divide the length
        # along each direction
        self.npx_x = int(self.lx // self.spacing)
        self.npx_y = int(self.ly // self.spacing)
        self.npx_z = int(self.lz // self.spacing)
        print("Grid dimensions are {}{}{}".format(self.npx_x,
                                                  self.npx_y, self.npx_z))

        # write config file
        self.cfg_file = self.name + '_void.cfg'
        with open(self.cfg_file, 'w') as cf:
            cf.write("lx = %f\n" % self.lx)
            cf.write("ly = %f\n" % self.ly)
            cf.write("lz = %f\n" % self.lz)
            cf.write("num_pixels = %i\n" % len(xx))
            cf.write("map_nx = %i\n" % self.npx_x)
            cf.write("map_ny = %i\n" % self.npx_y)
            cf.write("map_nz = %i\n" % self.npx_z)
            cf.write("corr_var_s = 0.23\n")
            cf.write("corr_l_perp = 5\n")
            cf.write("corr_l_para = 5\n")
            cf.write("pcg_tol = 1.0e-5\n")
            cf.write("pcg_max_iter = 1000\n")

        # run the program on the data
        message1 = subprocess.run(['./dachshund.exe', self.cfg_file],
                                  stdout=sys.stdout)
        print(message1)

        self.map_file = self.name + '_map.bin'
        message2 = subprocess.run(['mv', 'map.bin', self.map_file],
                                  stdout=sys.stdout)
        print(message2)

    def plot_interp(self, cuts):
        """ Plot the interpolated results """

        helper.viz_map(self.map_file, self.cfg_file, cuts, name=self.name,
                       pixel=self.chunk, sk_idx=self.sx)


if __name__ == '__main__':
    myobj = Interpolate('test', 'delta_transmission_RMplate.fits')

    myobj.get_data(skewers_perc=1., noise=0.1)
    myobj.process_box(xcuts=[-30, 30], ycuts=[-30, 30], zcuts=[3600, 3630])

    cuts = {'x': [], 'y': [], 'z': [1, 2, 3]}
    myobj.plot_interp(cuts)