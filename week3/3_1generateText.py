###########
# learning a language...
#
# read text from a file
# split the text into words
# go through the list, form tuples of words.
# look into the wordDict - if they have not occured yet...
#   - create another wordDict for all words that follow after that tuple
#   - put the following word into the wordDict
# ... if the tuple has already occured, skip creation otherwise do the same


# read text from a file
with open ("Frankensteinpg84.txt", "r") as myfile:
    text=myfile.read().replace('\n', ' ').replace('\r', '')
    
# split the text into words
wordList=text.split()

# go through the list, form tuples of two words.

wordDict= {} # create an empty dictionary

for i in range(len(wordList)-1):
    curWord=wordList[i]
    nextWord=wordList[i+1]
    if curWord not in wordDict:
        wordDict[curWord]={0:0} # put an empty dictionary into into the entry of the word
    if nextWord not in wordDict[curWord]:
        wordDict[curWord][nextWord]=0
    # now we can count that word combo...
    wordDict[curWord][nextWord]+=1
    wordDict[curWord][0]+=1 # for the total count of all words

########## speaking the language
# start with a few words.
# look into the wordDict of known tuples...
#  see how many entries the dictionary of following words has - draw a random one from them
# add the new word to the text and start over

import random

nWords=100
generatedWords=["I"]
# repeat the process until we have come up with enough words
for i in range(nWords):
    lastWord=generatedWords[-1]#in python lists, "-1" means "the last"
    associations=wordDict[lastWord]
    #select a random word from the associations
    generatedWords.append(list(associations.keys())[random.randrange(1,len(associations))])

print (" ".join(generatedWords))


