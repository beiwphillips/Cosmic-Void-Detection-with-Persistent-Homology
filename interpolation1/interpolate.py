#!/usr/bin/env python


import sys
import math
import subprocess
import timeit
import pickle
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from plotly.offline import plot
import plotly.graph_objs as go


class Interpolate:
    def __init__(self, filepath=None, sample_size=None, sample_rate=None, noise=0):
        if filepath is None:
            filepath = sys.argv[1]
        self.data = fits.open(filepath)
        self.Q = None
        self.p = None

        self.get_data(sample_size, sample_rate, noise)

    def get_data(self, sample_size, sample_rate, noise, viz=False):
        print('******READING******')
        length = len(self.data)
        if sample_size is not None:
            idx = np.random.choice(np.arange(1, length), size=sample_size, replace=False)
        elif sample_rate is not None:
            idx = np.random.choice(np.arange(1, length), size=int((length-1)*sample_rate), replace=False)
        else:
            raise ValueError()

        x_list, y_list, z_list, v_list = [], [], [], []
        for i in idx:
            x, y, z, v = self.__read_line(self.data[i])
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

        print('Read {} points from the file'.format(self.p.shape[0]))

    def __read_line(self, line):
        ra = (line.header['RA']) * np.pi / 180
        dec = (line.header['DEC']) * np.pi / 180
        rcomov = line.data['RCOMOV']
        value = line.data['DELTA_T']

        x = np.sin(dec) * np.cos(ra) * rcomov
        y = np.sin(dec) * np.sin(ra) * rcomov
        z = np.cos(dec) * rcomov
        v = value

        return x, y, z, v

    def __get_matrix(self):
        print('******CALCULATING ROTATION MATRIX********')
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

        print('Rotation Matrix: ')
        print(Q)

        self.Q = Q

    def process_cube(self):
        self.dachshund(self.p, 0)

    def process_points(self, n_points=10000, iterate_whole_cube=False, viz=False):
        print('******PROCESSING WITH {} points******'.format(n_points))
        z = self.p[:,2]
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
            self.dachshund(self.p[idx], cnt)
            start = end
            cnt += 1
            if not iterate_whole_cube:
                if viz:
                    self.visualize_interpolation(0, self.p[idx, 0], self.p[idx, 1], self.p[idx, 2])
                    self.visualize3d(self.p[idx, 0], self.p[idx, 1], self.p[idx, 2])
                break
        print('******FINISHED******')

    def dachshund(self, chunk, i):

        # write binary file
        chunk.tofile('pixel_data.bin')

        # write numpy file
        np.save('pixel_data.npy', chunk)

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

        message1 = subprocess.run(['./dachshund.exe', 'void.cfg'], stdout=sys.stdout)
        print(message1)
        newname = 'map{}.bin'.format(i)
        message2 = subprocess.run(['mv', 'map.bin', newname], stdout=sys.stdout)
        print(message2)

    def visualize_interpolation(self, i, x, y, z):
        print(z)
        npx_x = int(np.floor((x.max() - x.min()) / 2.))
        npx_y = int(np.floor((y.max() - y.min()) / 2.))
        npx_z = int(np.floor((z.max() - z.min()) / 2.))
        gs = (npx_x, npx_y, npx_z)

        xx = np.floor((x - x.min()) / 2)
        yy = np.floor((y - y.min()) / 2)
        zz = np.abs((z - z.min()) / 2 - z[i]) ** 2 / 0
        print(zz)
        plt.scatter(xx, yy, s=zz, c='black')

        m = np.fromfile('map{}.bin'.format(i))
        m_grid = m.reshape(gs)
        plt.imshow(m_grid[:, :, 1])
        cbar = plt.colorbar()
        cbar.set_label('delta', rotation=270)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def visualize3d(self, x, y, z):
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.5
            )
        )
        fig = go.Figure(data=[trace])
        plot(fig)


if __name__ == '__main__':
    interpolate = Interpolate('delta_transmission_RMplate.fits', sample_rate=1, noise=0.1)
    interpolate.process_one(viz=True)
