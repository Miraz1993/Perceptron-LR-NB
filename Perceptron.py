from os import listdir
from os.path import isfile, join
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import sys
import math

def vocabWords(classWiseFileList):
    uniqueVocabWords = set()
    for flForEachClass in classWiseFileList:
        for filePaths in flForEachClass:

            file = open(filePaths, encoding="utf8",errors='ignore')

            data = file.read().lower()

            words = word_tokenize(data)
            #
            wordsFiltered = [w for w in words ]

            for x in wordsFiltered:
                uniqueVocabWords.add(x)
    return uniqueVocabWords

def checkFile(filePath,uniqueVocabWords):
    file = open(filePath, encoding="utf8", errors='ignore')
    returnArray=[]
    data = file.read().lower()

    words = word_tokenize(data)
    for word in uniqueVocabWords:
        if word not in words:
            returnArray.append(0)
        else:
            returnArray.append(1)
    return returnArray

def outputCalc(x,w,t):
    op=np.matmul(x,w)

    if(op>0):
        return 1,t==1
    else:
        return -1,t==-1

def updatingWeight(eeta,t,o,x,weights):
    for i in range(len(x)):
        delw = eeta * (t - o) * x[i]
        weights[i] = weights[i] + delw
    return weights

def percTraining(input,test,weights,lCount):
    v = np.ones((len(input), 1))

    input = np.c_[v, input]
    row=len(input)
    eeta=0.001
    col=len(input[0])

    for j in range(lCount):

        correctP=0
        for l in range(row):
            t=test[l]
            x=input[l]

            o,val=outputCalc(x,weights,t)
            if(val==False):

                weights=updatingWeight(eeta,t,o,x,weights)
                #o, val = outputCalc(x, weights, t, l)


            else:
                correctP+=1



    return weights
def accuracyCheck(data,testClass,weights):

    data=[1]+data
    total=0
    total=np.matmul(data,weights)
    if(total>0):
        val=1
    else:
        val=-1


    return val==testClass

def dataMatching(data1,data2):
    row=len(data1)
    col=len(data1[0])
    for i in range(row):
        for j in range(col):
            if(data1[i][j]!=data2[i][j]):
                print("false")

def getMatrix(uniqueVocabWords,classWiseFileList,nameOfClasses,countDocs):

    data=[[0 for x in range(len(uniqueVocabWords))]for y in range(countDocs)]
    testClass=[0 for y in range(countDocs)]
    j=0
    for i in range(len(nameOfClasses)):
        filesForClass=classWiseFileList[i]


        for file in filesForClass:
            checkArr=checkFile(file,uniqueVocabWords)

            data[j]=checkArr
            if(i==0):
                testClass[j]=-1
            else:
                testClass[j] = 1
            j=j+1


    return data,testClass










nameOfClasses = []
classWiseTrainingFileList = []
classWiseValidationFileList = []
classWiseFileList = []
trainingSetPath=sys.argv[1]

#C:\Users\miraz\Downloads\hw2_train\train
#C:\Users\miraz\Downloads\enron1_train (1)\enron1
#C:\Users\miraz\Desktop\OwnDataSet\train
#C:/Users/miraz/Downloads/hw2_train/train/
#C:\Users\miraz\Downloads\enron1_train (1)\enron1\train
wc={}
v_dict={}
#stopWords = set(stopwords.words('english'))
allTrainClassPaths = listdir(trainingSetPath)

countTest=0
countTrain=0
countValidation=0
totCount=0
for c in allTrainClassPaths:
    nameOfClasses.append(c)
    newPath=trainingSetPath+c+"/"
    items=listdir(newPath)
    trainingSize=round(len(items)*0.7)
    validationSize=round(len(items)*0.3)
    trainingItems=items[0:trainingSize]
    validationItems=items[trainingSize:]
    trainingPath=[newPath+x for x in trainingItems]
    validationPath = [newPath + x for x in validationItems]
    allPath=[newPath + x for x in items]
    countTrain=countTrain+len(trainingPath)
    countValidation = countValidation + len(validationPath)
    totCount=totCount+ len(allPath)
    classWiseTrainingFileList.append(trainingPath)
    classWiseValidationFileList.append(validationPath)
    classWiseFileList.append(allPath)

uniqueVocabWords=vocabWords(classWiseTrainingFileList)
noOfWords=len(uniqueVocabWords)


input,testClass=getMatrix(uniqueVocabWords,classWiseTrainingFileList,nameOfClasses,countTrain)



weights=[0 for x in range(len(uniqueVocabWords)+1)]
itr=[5,10,15,20]
best=0
bestItr=-1
for item in itr:
    correctCounter = 0
    weights = [0 for x in range(len(uniqueVocabWords) + 1)]

    newWeights=percTraining(input,testClass,weights,item)
    data, test = getMatrix(uniqueVocabWords, classWiseValidationFileList, nameOfClasses, countValidation)

    for i in range(len(data)):
        val = accuracyCheck(data[i], test[i], newWeights)

        if (val):
            correctCounter = correctCounter + 1

    accuracy = float(correctCounter) / float(len(data))
    if(accuracy>best):
        best=accuracy
        bestItr=item

print("The best value for no of iterations is ",bestItr)

################
input,testClass=getMatrix(uniqueVocabWords,classWiseFileList,nameOfClasses,totCount)
weights=[0 for x in range(len(uniqueVocabWords)+1)]
newWeights=percTraining(input,testClass,weights,bestItr)

###############

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

data,test=getMatrix(uniqueVocabWords,testclassWiseFileList,nameOfClasses,countTest)


for i in range(len(data)):
    val=accuracyCheck(data[i],test[i],newWeights)

    if(val):
        correctCounter=correctCounter+1

accuracy = float(correctCounter)/float(len(data))
print("Final Accuracy: " + str(round(accuracy*100,2))+'%')