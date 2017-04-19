# an example of how to plot data



# load data
import pandas # use the pandas "Python Data Analysis Library"
data = pandas.read_csv("iris.txt", sep=" ") #read data from a file, columns are separated by " "


# import plotting functionality
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#In order to plot into separate windows, go to
#Tools -> preferences -> Ipython console -> graphics and selected backend

#plot data in 2d
#first, we need to convert those cathegory names to numbers. 
data.speciesCodes=data.Species.astype('category').cat.codes # from http://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers
plt.figure(1) # make a new plot window
plt.scatter(data.iloc[:, 0], data.iloc[:, 1],c=data.speciesCodes)

#plot data in 3d
fig = plt.figure(2) #create an empty figure
ax = Axes3D(fig)   # create a 3d space in the figure
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]) #show the measured items as dots "scatterplot"

# let's use different colors for different species:
ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],c=data.speciesCodes) #show the measured items as dots "scatterplot"
#add some descriptions to the axes
ax.set_xlabel(data.columns[0])
ax.set_ylabel(data.columns[1])
ax.set_zlabel(data.columns[2])