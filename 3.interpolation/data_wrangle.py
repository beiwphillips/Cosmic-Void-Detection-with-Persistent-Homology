import numpy as np
import fitsio


def dataWrangle(fits_file, save=True, name_tag="point_data", get_center=False):
    """ Extract relevant data from fits file
    Need to run only once!!!
    """
    fits_obj = fitsio.FITS(fits_file)

    # the first HDU is always an image - doesn't contain data
    n_skewers = len(fits_obj) - 1
    print("Total number of skewers are %d" % n_skewers)

    # Create arrays to store the data
    if get_center:
        ra_arr, dec_arr = [], []
    xx, yy, zz = [], [], []
    ff = []

    for i in range(1, len(fits_obj)):
        hdr = fits_obj[i].read_header()

        ra = np.deg2rad(hdr['RA'])
        dec = np.deg2rad(hdr['DEC'])

        if get_center:
            ra_arr.append(ra)
            dec_arr.append(dec)

        tb_data = fits_obj[i].read()
        r, field = tb_data['RCOMOV'], tb_data['DELTA_T']

        # Spherical to cartesian co-ordinates
        x = np.sin(dec) * np.cos(ra) * r
        y = np.sin(dec) * np.sin(ra) * r
        z = np.cos(dec) * r

        xx = np.hstack((xx, x))
        yy = np.hstack((yy, y))
        zz = np.hstack((zz, z))
        ff = np.hstack((ff, field))

    if save:
        np.savetxt(name_tag + ".dat", [xx, yy, zz, ff])
    if get_center:
        return np.mean(ra_arr), np.mean(dec_arr)


def celestial_rot_matrix(ra_point, dec_point, is_rad=True):
    """ Obtain a rotation matrix to rotate any set of
    cartesian points in such a way that the polar axes
    points along the (ra_point, dec_point) direction
    """
    if not is_rad:
        ra_point = np.deg2rad(ra_point)
        dec_point = np.deg2rad(dec_point)

    # Vector around which we rotate
    k_perp = np.array([-np.sin(ra_point), np.cos(ra_point), 0])

    # Rotation direction - Cross product matrix
    K = np.array([
        [0, -k_perp[2], k_perp[1]],
        [k_perp[2], 0, -k_perp[0]],
        [-k_perp[1], k_perp[0], 0]
        ])

    # ADOPTED FROM: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # Rotation Matrix to rotate the points vectors
    Q = np.eye(3) - np.sin(dec_point) * K +\
        (1 - np.cos(dec_point)) * np.matmul(K, K)

    return Q


def rot_points(data_file, ra_point, dec_point, viz=True):
    """ Rotate the points and visualize

    Parameters:
        data_file: input file containing the cordiantes and field values
        ra_point: central RA of the plate
        dec_point: central DEC of the plate
        offset: offset between the map (starts at 0) and pixel data
                Positive offset is preferred.
    """
    if data_file is None:
        data_file = "point_data.dat"

    *XX, ff = np.loadtxt(data_file)

    # rotate the points using Rotation matrix
    rot_matrix = celestial_rot_matrix(ra_point, dec_point)
    x, y, z = rot_matrix.dot(XX)


