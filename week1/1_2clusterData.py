# an example of how to plot data
# import plotting functionality
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data
import pandas # use the pandas "Python Data Analysis Library"
data = pandas.read_csv("iris.txt", sep=" ") #read data from a file, columns are separated by " "
data.speciesCodes=data.Species.astype('category').cat.codes # from http://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers

#import clustering functionality
import numpy
import sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(data.iloc[:,0:2])# apply "k-Means" Algorithm on first 4 cols
# see animation here:http://shabal.in/visuals/kmeans/1.html


# let's see how those decisions are actually made... 
# (from: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py)
#let's pick two dimensions of the data that we work with:

xData=data.iloc[:,0]
yData=data.iloc[:,1]

# we make a grid of euqually placed "probe points" to see what would be put under each category
stepsize=0.005
xMesh,yMesh=numpy.meshgrid(numpy.arange(xData.min(), xData.max(), stepsize), numpy.arange(yData.min(), yData.max(), stepsize))


# Obtain labels for each point in mesh. Use last trained model.
surfaceLabels = kmeans.predict(numpy.c_[xMesh.ravel(), yMesh.ravel()])


#plot an image with colors that tell us what a point would be labeled like:
surfaceLabels = surfaceLabels.reshape(xMesh.shape) 

plt.figure() # make a new plot window

# show the predictions of our dummypoints as colors of a surface
plt.imshow(surfaceLabels, interpolation='nearest',
           extent=(xMesh.min(), xMesh.max(), yMesh.min(), yMesh.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

#plot the original datapoints color them according to their label
plt.scatter(xData, yData,c=data.speciesCodes)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)


#numpy.linalg.norm(kmeans.cluster_centers_[0]-data.iloc[0,0:2]) # distance betwene two points

distsToCenters=sklearn.metrics.pairwise_distances(kmeans.cluster_centers_,data.iloc[:,0:2]) #distances between points and cluster centers
numpy.argmin(distsToCenters[0,:]) #who is closest to the center?
