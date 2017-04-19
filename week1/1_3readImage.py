## from http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
# load data

#load all kinds of libraries...
from skimage.color import rgb2yuv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D # import 3d-plotting functionality

n_colors = 4

# Load the Summer Palace photo
image = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
image = np.array(image, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array. with pixels as rows and color components as columns
w, h, d = original_shape = tuple(image.shape)
assert d == 3 # make sure it's a color image.
image_array = np.reshape(image, (w * h, d))

#Perform all future operations on a small sub-sample of the data to save some time
image_array_sample = shuffle(image_array, random_state=0)[:1000] # take 1000 random pixels to save some time

# Display original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(image)

#plot data in 3d
fig = plt.figure(2) #create an empty figure
ax = Axes3D(fig)   # create a 3d space in the figure
ax.scatter(image_array_sample[:, 0], image_array_sample[:, 1],image_array_sample[:, 2],c=image_array_sample) #show the measured items as dots "scatterplot"


#Let the clustering try to determine different regions in the image
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample) 
labels=kmeans.predict(image_array) # get clusters of all pixels

# plot the image with it's cluster indices as colors
plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(np.reshape(labels,(w,h)))


#plot clustered data in 3d
fig = plt.figure(4) #create an empty figure
ax = Axes3D(fig)   # create a 3d space in the figure
ax.scatter(image_array_sample[:, 0], image_array_sample[:, 1],image_array_sample[:, 2],c=kmeans.labels_) #show the measured items as dots "scatterplot"


# lets try a better clustering:
from sklearn import mixture
dpgmm = mixture.BayesianGaussianMixture(n_components=5,covariance_type='full').fit(image_array_sample)

#plot clustered data in 3d
fig = plt.figure(5) #create an empty figure
ax = Axes3D(fig)   # create a 3d space in the figure
ax.scatter(image_array_sample[:, 0], image_array_sample[:, 1],image_array_sample[:, 2],c=dpgmm.predict(image_array_sample)) #show the measured items as dots "scatterplot"


# plot the image with it's cluster indices as colors
plt.figure(6)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(np.reshape(dpgmm.predict(image_array),(w,h)))
