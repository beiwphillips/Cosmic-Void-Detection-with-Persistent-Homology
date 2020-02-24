import numpy as np
# import matplotlib.pyplot as plt


x = np.arange(0, 24)[:, None, None]
y = np.arange(0, 24)[:, None]
z = np.arange(0, 150)


amp_one = 1.5
amp_two = 0.8
corr = 2  # in h^{-1} Mpc

void_one = amp_one * np.exp(- (1. / corr ** 2) * ((x - 12) ** 2
                            + (y - 12) ** 2 + (z - 50) ** 2))

void_two = amp_two * np.exp(- (1. / corr ** 2) * ((x - 10) ** 2
                            + (y - 4) ** 2 + (z - 130) ** 2))

# total field
field = void_one + void_two

# pad with all zeros
field[field < 10 ** -3] == 0

# plt.imshow(field[:, :, 130], cmap=plt.cm.seismic)
# plt.show()
# save to file
np.savetxt("two_lonely_voids.txt", np.ravel(field))

# # to read back just use
# data = np.loadtxt("two_lonely_voids.txt")
# data = data.reshape((len(z), len(y), len(x)))

# # SHAPE OF FIELD IS (LEN(X), LEN(Y), LEN(Z))

# # visualization
# plt.figure(figsize=(8, 8))
# plt.imshow(field[180, :, :])


# mayavi visualization
from mayavi import mlab


mlab.pipeline.volume(mlab.pipeline.scalar_field(field))


# mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(field),
#                                  plane_orientation='z_axes',
#                                  slice_index=10,
#                                  colormap='seismic'
#                                  )

mlab.colorbar(title='delta', orientation='vertical')
mlab.outline()
mlab.show()
