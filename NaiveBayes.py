from os import listdir
from os.path import isfile, join
from collections import Counter


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import sys
import math
import  random as r


def gettingWordCount(pathList):
    wc = Counter()
    for name in pathList:

        file = open(name, "r",encoding="utf8",errors='ignore')

        data = file.read()
        r_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(r_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        wc.update(wordsFiltered)
    return wc


def conditionalProbCalc(className,wordCountInEachClass):
    wordsInClass=len(wordCountInEachClass)
    for word in uniqueVocabWords:
        if word in wordCountInEachClass.keys():
            prob_class = (wordCountInEachClass[word] + 1) / (wordsInClass + noOfWords)
        else:
            prob_class = (0 + 1) / (wordsInClass + noOfWords)


        if nameOfClasses.index(className) == 0:
            v_dict[word] = {className:prob_class }
        else:
            v_dict[word][className]=prob_class




def vocabWords(classWiseFileList):
    uniqueVocabWords = set()
    for flForEachClass in classWiseFileList:
        for filePaths in flForEachClass:

            file = open(filePaths, encoding="utf8",errors='ignore')

            data = file.read().lower()

            words = word_tokenize(data)
            wordsFiltered = [w for w in words if w not in stopWords]
            for x in wordsFiltered:
                uniqueVocabWords.add(x)
    return uniqueVocabWords

def NBTraining(countTrain,classWiseFileList, vocab,NameOfClasses):

    for i in range(len(NameOfClasses)):
        dCount = len(classWiseFileList[i])
        prior[i] = float(dCount) / float(countTrain)
        wc[i] = gettingWordCount(classWiseFileList[i])
        conditionalProbCalc(NameOfClasses[i], wc[i])

def testNB(NameOfClasses,v_dict,eachFilePath,prior):
    file = open(eachFilePath, "r",encoding="utf8",errors='ignore')
    data = file.read()

    r_data = re.sub('[^a-zA-Z\n]', ' ', data)
    words = word_tokenize(r_data)
    wordsFiltered = [w for w in words if w not in stopWords]
    score = [0 for x in range(len(NameOfClasses))]
    for i in range(len(NameOfClasses)):
       score[i] = math.log(prior[i])
       for word in wordsFiltered:
            if word in v_dict.keys():
               classvalue=NameOfClasses[i]
               score[i] += math.log(v_dict[word][classvalue])

    return NameOfClasses[score.index(max(score))]

nameOfClasses = []
classWiseFileList = []

trainingSetPath=sys.argv[1]
#"C:/Users/miraz/Downloads/hw2_train/train/"
wc={}
v_dict={}
stopWords = set(stopwords.words('english'))
allTrainClassPaths = listdir(trainingSetPath)

countTest=0
countTrain=0
for c in allTrainClassPaths:
    nameOfClasses.append(c)
    newPath=trainingSetPath+c+"/"
    listdir(newPath)
    path=[newPath+x for x in listdir(newPath)]
    countTrain=countTrain+len(path)
    classWiseFileList.append(path)
#ob=NaiveBayes()
prior = [0 for x in range(len(nameOfClasses))]

uniqueVocabWords=vocabWords(classWiseFileList)
noOfWords=len(uniqueVocabWords)
NBTraining(countTrain,classWiseFileList,uniqueVocabWords,nameOfClasses)



########################
pathForTestingDataSet=sys.argv[2]
nameOfTestClasses = []
testclassWiseFileList = []
for c in nameOfClasses:

    nameOfTestClasses.append(c)
    newPath = pathForTestingDataSet + c + '/'

    path = [newPath + x for x in listdir(newPath)]
    countTest = countTest + len(path)
    testclassWiseFileList.append(path)

correctCounter=0

for eachClassIndex in range(len(nameOfTestClasses)):
    for eachfile in testclassWiseFileList[eachClassIndex]:
        predictClass = testNB(nameOfTestClasses,v_dict,eachfile,prior)
        if(predictClass == nameOfTestClasses[eachClassIndex]):
            correctCounter+=1

accuracy = float(correctCounter)/float(countTest)
print("Accuracy for the test set is: " + str(round(accuracy*100,2))+'%')
#for()