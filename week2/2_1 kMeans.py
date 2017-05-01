import pandas
import numpy
import random
import matplotlib.pyplot as plt 
from matplotlib import collections  as mc
from sklearn.cluster import KMeans

data = pandas.read_csv('iris.txt',sep=' ') # read the table with some library magic
data.speciesCodes=data.Species.astype('category').cat.codes # from http://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers

npdata=data.iloc[:,0:4].values # select first four columns and convert them to a numpyarray
print(npdata.shape)
#plot the original datapoints color them according to their label


plt.figure() # make a new plot window
plt.scatter(npdata[:,0], npdata[:,1],c=data.speciesCodes) # draws data as a collection of dots

numClusters=4 # how many clusters do we want?

indexList=[] # initialize an empty list of indices
# let's make a list of random numbers
for i in range(numClusters):
    randomIndex=random.randint(0,npdata.shape[0])
    #find out if the index is in the list already
    # =>find out by using "listName.count(thingToFind)"
    # continue drawing new numbers until we have found one that is not
    # => use "while" for that
    while indexList.count(randomIndex)>0:
        randomIndex=random.randint(0,npdata.shape[0]-1)

    
    indexList.append(randomIndex)
    print (randomIndex)
### the short way....
#random.sample(range(npdata.shape[0]),numClusters)


print( indexList)
startClusterCenters=npdata[indexList,:]


for kmeansInter in range(100):
    #make a space for the new cluster centers:
    newClusterCenters= numpy.zeros(startClusterCenters.shape)
    clusterCounts=numpy.zeros(numClusters) # here we count how many we found
    #################################################
    # find out which clustercenter is closest to each flower/datapoint
    for dpIndex in range(npdata.shape[0]): # repeat for all datapoints
        curPoint=npdata[dpIndex,:]
        #numpy.linalg.norm(a-b) # this is how we get distance between a and 
        distToAllClusters=[]
        # this is how we get distance between datapoint and all clustercenters:
        for clustIndex in range(numClusters): # repeat for all clusters
            distToAllClusters.append(numpy.linalg.norm(curPoint-startClusterCenters[clustIndex,:]))
        
        closestClusterIndex=numpy.argmin(distToAllClusters)
        # now we know wich cluster is closest... now what?
        
        # add the datapoint to the new cluster center
        newClusterCenters[closestClusterIndex,:]+=curPoint
        # add one to the number of points in the cluster
        clusterCounts[closestClusterIndex]+=1
        
    #divide the summed up datapoints by the number of them to get the mean
    for clustIndex in range(numClusters): # repeat for all clusters
        newClusterCenters[clustIndex,:]/=clusterCounts[clustIndex]
        #plot a line from each old cluster center to the new position
        plt.plot((startClusterCenters[clustIndex, 0],newClusterCenters[clustIndex, 0]), (startClusterCenters[clustIndex, 1], newClusterCenters[clustIndex, 1]))
    # plot the old cluster centers in one color
    plt.scatter(startClusterCenters[:, 0], startClusterCenters[:, 1],
                marker='x', s=169, linewidths=3,
                color='b', zorder=10)
    # and the new ones in a different one
    plt.scatter(newClusterCenters[:, 0], newClusterCenters[:, 1],
                marker='x', s=169, linewidths=3,
                color='r', zorder=10)
    
    startClusterCenters=newClusterCenters # start next round with the value swe just got