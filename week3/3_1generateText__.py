###########
# learning a language...
#
# read text from a file
# split the text into words
# go through the list of words.
# look into the wordDict - if they have not occurred yet...
#   - create another wordDict for all words that follow after that first word
#   - put the following word into the wordDict
# ... if the tuple has already occured, skip creation otherwise do the same


# word1 --> {wordA: 10, wordB:1, wordC:5}
with open ("Frankensteinpg84.txt", "r") as myfile:
    text=myfile.read().replace('\n'," ").replace('\r',"")

#split the text into words
wordList= text.split();

#make a dictionary for the words
wordDict = {} 
wordCountDict= {} # for remembering the total word counts
for i in range (1, len(wordList)-1):
    # take two consecutive words from the wordlist
    wordBefore=wordList[i-1]
    curWord=wordList[i] 
    nextWord=wordList[i+1]
    # check if we have already encountered the first word:
    if (wordBefore,curWord) not in wordDict:
        wordDict[(wordBefore,curWord)]={}
        wordCountDict[(wordBefore,curWord)]=0 #  create a place where to count word occurrence
    # is the next word already added to the entry of the current word?
    # if not, create a space for counting the occurrences of this combination of words
    if nextWord not in wordDict[(wordBefore,curWord)]:
        wordDict[(wordBefore,curWord)][nextWord]=0 # put a zero under the entry for curword-> nextword
    wordDict[(wordBefore,curWord)][nextWord]+=1 # remember that we have seen this combination once
    wordCountDict[(wordBefore,curWord)]+=1 #  remember that we found the first word once

########## speaking the language
# start with a word.
# look into the wordDict...
#  see how many entries the dictionary of following words has - draw a random one from them
# add the new word to the text and start over

import random

nWords=100 # how many words do we want to generate?
generatedWords =[list(wordDict.keys())[1000][0],list(wordDict.keys())[1000][1]] # add one word to the list to start with

for i in range(nWords):
    beforeLastWord=generatedWords[-2] #using negative idices, we can count from the end of the list
    lastWord=generatedWords[-1] #using negative idices, we can count from the end of the list
    
    
    associations= wordDict[(beforeLastWord,lastWord)]
    # draw next word according to it's frequency in the corpus
    potentialNextWords=list(associations.keys())
    randNum= random.randrange(0,wordCountDict[(beforeLastWord,lastWord)]) # draew a random number to determine the next word
    wordCountSum=associations[potentialNextWords[0]]# start with the counts of the first word in the dict
    curNextWordIndex=0
    while randNum>wordCountSum:
        curNextWordIndex+=1 # the random number was higher than the cumulated occurences up to this
        # so we go to the next word in the list
        wordCountSum+=associations[potentialNextWords[curNextWordIndex]]
        
    nextWord=list(associations.keys())[curNextWordIndex]
    generatedWords.append(nextWord)

print (" ".join(generatedWords)) #  concaternate all the words in the list, join them with a space

