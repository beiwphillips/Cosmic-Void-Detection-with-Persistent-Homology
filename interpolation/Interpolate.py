#!/usr/bin/env python
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_wrangle import celestial_rot_matrix
from astropy.io import fits

from plotly.offline import plot as go_plot
import plotly.graph_objs as go


class Interpolate:
    def __init__(self, filepath=None):
        if filepath is None:
            filepath = sys.argv[1]
        self.data_file = fits.open(filepath)

        self.rot_matrix = None
        self.pixel_data = None

        self.ra = None
        self.dec = None
        self.zq = None

        self.spacing = 4

        # self.get_matrix()
        # self.get_data(sample_size, sample_rate, noise)

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

        # concatenating together
        x = np.hstack(x_list)
        y = np.hstack(y_list)
        z = np.hstack(z_list)
        v = np.hstack(v_list)

        # rotating the points
        x, y, z = np.dot(self.rot_matrix, np.array([x, y, z]))

        # generating noise data
        n = np.ones_like(x) * noise

        # concatenating together
        self.pixel_data = np.vstack([x, y, z, n, v]).T

        print('Read {} points from the file'.format(len(x)))

    @staticmethod
    def visualize3d(pixel_data, plot_1d, plot_2d):
        if plot_1d:
            fig, ax = plt.subplots(ncols=3)
            for i in range(3):
                sns.distplot(pixel_data[:, i], ax=ax[i], kde=False)
        plt.show()

        if plot_2d:
            plt.figure()
            plt.scatter(pixel_data[:, 0], pixel_data[:, 1], s=0.1, c="k")
        plt.show()

        trace = go.Scatter3d(x=pixel_data[:, 0], y=pixel_data[:, 1],
                             z=pixel_data[:, 2], mode='markers',
                             marker=dict(size=1, opacity=0.5))
        fig = go.Figure(data=[trace])
        go_plot(fig)

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

        # shift the cordinates to the edge of the cube
        self.chunk = self.pixel_data[ixs]

        self.chunk[:, 0] -= self.chunk[:, 0].min() + 5
        self.chunk[:, 1] -= self.chunk[:, 1].min() + 5
        self.chunk[:, 2] -= self.chunk[:, 2].min() + 5

        # self.visualize3d(self.chunk, True, True)

        self.dachshund()

    def dachshund(self, offset=5, counter=0):
        # write binary file
        self.chunk.tofile('pixel_data.bin')

        # calculate values
        xx, yy, zz = self.chunk[:, :3].T

        # number of points in the grid
        self.lx = xx.max() - offset
        self.ly = yy.max() - offset
        self.lz = zz.max() - offset

        self.npx_x = int((xx.max() - offset) // self.spacing)
        self.npx_y = int((yy.max() - offset) // self.spacing)
        self.npx_z = int((zz.max() - offset) // self.spacing)
        print("dimensions of the grid are {}{}{}".format(self.npx_x,
                                                         self.npx_y, self.npx_z))

        # write config file
        cf = open("void.cfg", "w")
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
        cf.close()

        message1 = subprocess.run(['./dachshund.exe', 'void.cfg'], stdout=sys.stdout)
        print(message1)
        newname = 'map{}.bin'.format(counter)
        message2 = subprocess.run(['mv', 'map.bin', newname], stdout=sys.stdout)
        print(message2)

        self.visualize_interpolation(2)

    def visualize_interpolation(self, i, counter=0):
        gs = (self.npx_x, self.npx_y, self.npx_z)

        xline = np.linspace(0, self.lx, self.npx_x)
        yline = np.linspace(0, self.ly, self.npx_y)
        zline = np.linspace(0, self.lz, self.npx_z)

        ixs = np.where(np.abs(self.chunk[:, 3] - zline[i]) <= 2)[0]

        # plot the ith slice along z
        m = np.fromfile('map{}.bin'.format(counter))
        m_grid = m.reshape(gs)
        plt.pcolormesh(xline, yline, m_grid[:, :, i].T)
        cbar = plt.colorbar()
        cbar.set_label('delta', rotation=270)


        # overlay the data points
        plt.scatter(self.chunk[:, 0], self.chunk[:, 1], cmap=plt.cm.viridis)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()



if __name__ == '__main__':
    interpolate = Interpolate('../../delta_transmission_RMplate.fits')
    interpolate.get_locations()
    interpolate.get_matrix()

    interpolate.get_data(skewers_perc=1., noise=0.1)
    # interpolate.visualize3d(plot_1d=True, plot_2d=True)
    interpolate.process_box(xcuts=[-30, 30], ycuts=[-30, 30], zcuts=[3600, 3630])
