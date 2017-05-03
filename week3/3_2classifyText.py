########## identifying the speaker/topic
# look at each word independently
# see how often it has occurred in each of the known texts (word frequency)
# add "one" to each set to avoid zero probabilities for words that did not occur
import math
from sklearn.datasets import fetch_20newsgroups
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'sci.space',
]
# get data from the popular  "20newsgroups" dataset to play around

dataset = fetch_20newsgroups(subset='train', categories=categories,
                             shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

#the actual posts lie in dataset['data'] 
#the groups they were taken from are in dataset['target'] 


#let's go through all texts ...
wordDict= {} # create an empty dictionary where we will put our data
totalCounts=[0]*len(categories) #make space for storing the total numer of words of each cat.

for textIndex in range(len(dataset['data']) ):
    curText=dataset['data'][textIndex]
    curLabel=dataset['target'] [textIndex]
    # now let's go trough all the words in that message
    wordList=curText.split()
    for curWord in wordList:
        if curWord not in wordDict: #is it a word we haven't seen so far?
            wordDict[curWord]=[1]*len(categories) #put a "one" for each category into the dictionary
        wordDict[curWord][curLabel]+=1 # add a count to the text category where we found it
        totalCounts[curLabel]+=1 # add a count to the text category where we found it
        


# divide occurrence counts by total number of words to arrive at word frequencies
wordFrequencies={}

# add the dummy counts we initialized the counters with to the number of words per class
for i in range(len(categories)):
    totalCounts[i]+=len(wordDict)
for curWord in wordDict.keys():
    frequencies=[0]*3
    for i in range(len(categories)):
        frequencies[i]=wordDict[curWord][i]/totalCounts[i]
    wordFrequencies[curWord]=frequencies
    
#now for a new dataset...
# we go through each word
# look up the frequencies that this word had in the training sets
# multiply them all together (or rather use sums of logarithms, that is)
# compare the obtained values for all classes that were used for classification

testSet = fetch_20newsgroups(subset='test', categories=categories,
                             shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

testText=testSet['data'][0] #get the first text
logSums=[0]*len(categories) #make space for storing the total numer of words of each cat.
wordsInTestText=testText.split()
for curWord in wordsInTestText:
    #have we seen that word before?
    if curWord in wordFrequencies: 
        print (curWord+ str(wordDict[curWord]))
        for i in range (len(categories)):
           logSums[i]+=math.log(wordFrequencies[curWord][i])
bestCat=logSums.index(max(logSums))
print ("I think this is from "+dataset["target_names"][bestCat])


###in order to find out what made the difference, we can...
# for each class and word, calculate the contribution to the Kullback-Leibler divergence
# it is basically the difference of the log( frequencies) (which is what we added up before) times the 
# frequency of this word in the category 
wordImportances={}
for curWord in wordDict.keys():
    importances=[0]*3
    for i in range(len(categories)):
            for o in range(len(categories)):
                if i!=o:
                    importances[i]+=wordFrequencies[curWord][i]*math.log(wordFrequencies[curWord][i]/wordFrequencies[curWord][o])
    wordImportances[curWord]=importances

# let's sort the words by their importance and look at the first few...
uniqueWords=list(wordImportances.keys())
sortedFor0=sorted(uniqueWords,key=lambda word:-wordImportances[word][0])
sortedFor1=sorted(uniqueWords,key=lambda word:-wordImportances[word][1])
sortedFor2=sorted(uniqueWords,key=lambda word:-wordImportances[word][2])