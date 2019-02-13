from plotly.offline import plot
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualizeMap(np_file, map_file, cfg_file, cuts, width=2, prefix='test'):
    """
    Plot the mapped data

    Parameters:
    ----------
    map_file : binary file containing mapped data
    cfg_file : config file use in Dachshund
    cuts : dict containing the index of slices along each axes
           eg: {'x':[0, 2], 'y':[3, 4], 'z':[0, 1, 2]}
    pixel : file containing the pixel data
    sk_idx : skewer index of data-points
    width : width over which to avg the points in h^-1 Mpc

    Returns: None
    """

    # read numpy data
    np_data = np.load(np_file)

    # read map data
    map_data = np.fromfile(map_file)

    # read metadata from the config file
    with open(cfg_file) as f:
        fields = f.readlines()
        cfg = {}
        for field in fields:
            data = field.split('=')
            key = data[0].strip()
            val = float(data[1].strip())
            cfg[key] = val

    shape = (int(cfg['map_nx']), int(cfg['map_ny']), int(cfg['map_nz']))
    map_data = map_data.reshape(shape)

    # set limits for colorbar
    vmin, vmax = map_data.min(), map_data.max()

    npt_x, npt_y, npt_z = len(cuts['x']), len(cuts['y']), len(cuts['z'])
    lx, ly, lz = float(cfg['lx']), float(cfg['ly']), float(cfg['lz'])

    # the central values for the mapped pixels
    xline = (np.arange(shape[0]) + 0.5) * lx / shape[0]
    yline = (np.arange(shape[1]) + 0.5) * ly / shape[1]
    zline = (np.arange(shape[2]) + 0.5) * lz / shape[2]

    xx, yy, zz = np_data[:, :3].T
    vals = np_data[:, 4]
    sk_idx = np_data[:, -1]

    fig, ax = plt.subplots(nrows=int(np.ceil(npt_z / 3)), ncols=3)
    ax = np.atleast_2d(ax)

    for ct, ele in enumerate(cuts['z']):
        ii = (ct // 3, ct % 3)

        # plot the mapped data
        cbar = ax[ii].imshow(map_data[:, :, ele].T, origin="lower", vmin=vmin,
                             vmax=vmax, extent=(0, lx, 0, ly),
                             cmap=plt.cm.jet)

        ax[ii].set_title(r"$z = %.1f [h^{-1}\ Mpc]$" % zline[ele])

        # plot the data points
        ixs = (zz > zline[ele] - width) & (zz <= zline[ele] + width)

        # average over data points per skewer
        df = pd.DataFrame(np.array([xx[ixs], yy[ixs], vals[ixs]]).T,
                          index=sk_idx[ixs], columns=['x', 'y', 'v'])
        agg_df = df.groupby(df.index).agg(np.mean)

        ax[ii].scatter(agg_df['x'], agg_df['y'], c=agg_df['v'], vmin=vmin,
                       vmax=vmax, cmap=plt.cm.jet, edgecolor='k')

        if ii[1] == 0:
            ax[ii].set_ylabel(r'$y\ [h^{-1} Mpc]$')

        if ii[0] == int(np.ceil(npt_z / 3)) - 1:
            ax[ii].set_xlabel(r'$x\ [h^{-1} Mpc]$')

    # Common colorbar for all the subplots
    fig.colorbar(cbar, ax=ax.ravel().tolist(), orientation='horizontal')

    plt.savefig(prefix + 'plot.pdf')


def visualize3d(x, y, z):
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
    visualizeMap('pixel_data.npy', 'map.bin', 'void.cfg', cuts={'x': [], 'y': [], 'z': [1, 2, 3]},
                 width=5, prefix='')
