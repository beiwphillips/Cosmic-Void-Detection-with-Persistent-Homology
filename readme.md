# DETECTING COSMIC VOIDS WITH PERSISTENT HOMOLOGY

This project is about using persistent homology - a classic technique in topological data analysis, to detect cosmic voids and track their evolution from data cubes coming from Baryon Oscillation Spectroscopic Survey (BOSS) project of SDSS.

The raw data coming from BOSS is in the format of *.fits and organized as a set of line of sights (skewers). On each line of sights (skewers), there are sampled points which measure the density of a specific area in the space. 

We first interpolated the discrete raw data cube using wiener filter reconstruction and generated full data cubes. We then ran persistent homology to detect cosmic voids over the interpolated data cubes.

## DATA
1. *cube*: a demonstration cube of size 26*29*116
2. *small cubes*: a demonstration cube of size 14*13*7
3. *mock*: two testing cubes: stochastic and two-lonely-voids
4. *real*: a real cube
5. *real_wrongpara*: a real cube interpolated with wrong parameters
6. *realshuffled*: a set of cubes derived from the original real cube by shuffling the points from different skewers. Used as backgrounds.
7. *simulation*: a simulated cube
8. *simulation_wrongpara*: a simulated cube interpolated with wrong parameters
9. *simulationshuffled*: a set of cubes derived from the original simulated cube by shuffling the points from different skewers. Used as backgrounds.
10. *simulationdownsample*: a set of cubes derived from the original simulated cube by down-sampling the points from different skewers.

## CODE

### /1.data
Code to visualize the raw discrete data cube (fits files) to get familar with the data.

*tutorial_astropy.ipynb*
A tutorial for astropy.

*tutorial_plotly.ipynb*
A tutorial for ployly

*visualization1.ipynb*
*visualization2.ipynb*
*visualization3.ipynb*
*visualization_vikrant.ipynb*
Code for visualizing the raw discrete data cubes.

### /2.wiener
Code to get familiar with wiener filter reconstruction. This is just FYI if you are working on the computational topology side because this my require a lot of expertise in astrophysics and mathematics.

*wienerfilter_1d.ipynb*
A demonstration for interpolating discrete 1d data with wiener filter reconstruction manually.

*wienerfilter_2d.ipynb*
A demonstration for interpolating discrete 2d data with wiener filter reconstruction manually.

*wienerfilter_3d.ipynb*
A demonstration for interpolating discrete 3d data with wiener filter reconstruction using Dachshund.

### /3.interpolation
Code to run wiener filter reconstruction using Dachshund on the raw discrete data cubes (fits files) to get the interpolated data cubes (bin files) for voids finding with persistent homology. This just FYI if you are working on the computational topology side because this work is tedious and may require some expertise in astrophysics.

*interpolation_Dachshund.ipynb*
*interpolation_RegularGridInterpolator.ipynb*
*interpolation_ResultVisualization.ipynb*
Code to run wiener filter reconstruction using Dachshund and regular grid interpolation with Scipy over the discrete raw data cube and visualize the interpolated data cubes.

*data_wrangle.py*
*helper.py*
*interpolate.py*
*visualization.py*
Command line code to run wiener filter reconstruction using Dashshund written by Vikrant.

### /4.time
Code to collect and predict the time needed for running wiener filter reconstruction using Dachshund on different machines in SCI Institute. You don't have to re-run anything. The only message to take home is: kraken > lakota > chiron. Refer to the SCI website for details.

*rates_time.py*
*rates_time.sh*
*sizes_time.py*
*sizes_time.sh*
*rates_time_cluster.py*
*sizes_time_cluster.py*
*time.conf*
*time.slurm*
Code to collect the time spent to run wiener filter reconstruction with different parameters on sampling rate and cube size.

*Time1.ipynb*
*Time2.ipynb*
*Time3.ipynb*
Code to make predictions based on the collected data.

### /5.finding
Code to find cosmic voids inside interpolated data cubes (bin files). This is the core code for this project.

*pixel2d.py*
The class which defines a 2-dimensional pixel.

*pixel3d.py*
The class which defines a 3-dimensional pixel.

*topologicalunionfind.py*
The class which runs the union-find algorithm over any types of components. An additional parameter - time - is required in the "add" and "union" operations which captures the time new connected components are generated. This enables topological features (persistence) in this class.

*voidsfinding.py*
The class which runs topological union-find or Perseus program to find voids in a 2-dimensional or 3-dimensional data cube, return or plot the history of the evolution of voids.

*main.py*
The command line interface to run topological union-find to find voids in data cubes and return the results.

*prod_VoidsFindingToy2D.ipynb*
A script which runs the program over a toy 2D slice.

*prod_VoidsFindingToy3D.ipynb*
A script which runs the program over a toy 3D cube.

*prod_VoidsFinding2D.ipynb*
A script which runs the program over a 2D slice of an example cube.

*prod_VoidsFinding3D.ipynb*
A script which runs the program over a 3D example cube.

*prod_VoidsFindingReal.ipynb*
A script which runs the program over a real astronaumical cube.

*prod_VoidsFindingSimulation.ipynb*
A script which runs the program over a simulated astronaumical cube.

*prod_Mock.ipynb*
A script which runs the program over two mock cubes: two lonely voids and stochastic. This can be used as verification.

*prod_History.ipynb*
A script which runs the program over multiple cubes and output the evolution history JSON file. The JSON file is used for 3D rendering.

*prod_Paper.ipynb*
A script which runs the program and produces refined plots for publications.

*explore_\*.ipynb*
Scripts used to explore the cubes and test the program.

*\*.txt*
Intermediate files generated for or by Perseus.

### 6.lynne
Code from Lin Yan for running persistent homology using Perseus.

## PAPERS
### /papers
Publications for reference

## EXTERNAL PACKAGES
### /dachshund

https://github.com/caseywstark/dachshund

A package to run wiener filter reconstruction to interpolate the data cubes. This package is needed only for the interpolation step.

### /perseus

http://people.maths.ox.ac.uk/nanda/perseus/index.html

A package to run persistent homology over different simplices. We only used this package for sanity check. This package is needed if you want to run persistent homology with Perseus.
