# an example of how to load data from a text file

# load data
import pandas # use the pandas "Python Data Analysis Library"
data = pandas.read_csv("iris.txt", sep=" ") #read data from a file, columns are separated by " "


#print some of the data as text:
print(data) # it can be so simple...


print(data.loc[0,'Species']) #Selection by column name: get the "species" of the first plant in the set
# with ".loc", you can identify oclumns/rows by their names, but it is somewhat slow

print(data.Species) # due to python magic, we can even do this!

print(data.iloc[0,4]) #Selection by column number: get the "species" of the first plant in the set
# "the "i" in "iloc" stands for "index"

print(data.iloc[0,:]) # get the complete first row of the dataset
# ":" means "all of them" - the keyword is "slicing"

print(data.iloc[:,0]) # get the complete first column of the dataset


#extract some specific datapoints
data.loc[data.Species=='setosa',:] # extract all datapoints with the speciesname "setosa"
