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

            wordsFiltered = [w for w in words if w not in stopWords]

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

def calculatingSigmoid(x,weights):
    total=0
    for i in range(len(x)):
        total+=x[i]*weights[i]
    try:
        expValue=math.exp(total)
        sigm=expValue/(expValue+1)
    except:
        sigm=1
    return sigm
def gradientArray(input,testClass,weights):

    grad = [0 for x in range(len(input))]
    for l in range(len(input)):

        prob = calculatingSigmoid(input[l], weights)

        grad[l]=(testClass[l] - prob)

    return grad
def gradientValue(grad,input,i):
    gradValue=0
    for l in range(len(input)):
        gradValue+=input[l][i]*grad[l]
    return gradValue


def gradientAscent(input,testClass,weights,lamb):
    v = np.ones((len(input), 1))

    input = np.c_[v, input]


    for j in range(20):

        temp = weights
        lRate=0.01

        grad = gradientArray(input, testClass, weights)
        for i in range(len(weights)):
            gradValue=gradientValue(grad,input,i)
            temp[i] = temp[i] + (lRate * gradValue)-(lRate*lamb*temp[i])
        weights = temp


    return weights


def getMatrix(uniqueVocabWords,classWiseFileList,nameOfClasses,countDocs):

    data=[[0 for x in range(len(uniqueVocabWords))]for y in range(countDocs)]
    testClass=[0 for y in range(countDocs)]
    j=0
    for i in range(len(nameOfClasses)):
        filesForClass=classWiseFileList[i]


        for file in filesForClass:
            checkArr=checkFile(file,uniqueVocabWords)

            data[j]=checkArr
            testClass[j]=i
            j=j+1


    return data,testClass
def accuracyCheck(data,testClass,weights):
    data=[1]+data
    total=0
    for i in range(len(weights)):
        total+=data[i]*weights[i]
    if(total>0):
        val=1
    else:
        val=0


    return val==testClass
nameOfClasses = []
classWiseTrainingFileList = []
classWiseValidationFileList = []
classWiseFileList = []
trainingSetPath=sys.argv[1]
#C:\Users\miraz\Downloads\hw2_train\train
#C:\Users\miraz\Downloads\enron1_train (1)\enron1
wc={}
v_dict={}
stopWords = set(stopwords.words('english'))
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
lamb=[0]
best=0
bestLamb=-1
for item in lamb:
    correctCounter = 0
    weights = [0 for x in range(len(uniqueVocabWords) + 1)]
    newWeights=gradientAscent(input,testClass,weights,item)
    data, test = getMatrix(uniqueVocabWords, classWiseValidationFileList, nameOfClasses, countValidation)

    for i in range(len(data)):
        val = accuracyCheck(data[i], test[i], newWeights)

        if (val):
            correctCounter = correctCounter + 1

    accuracy = float(correctCounter) / float(len(data))
    if(accuracy>best):
        best=accuracy
        bestLamb=item

############################
print("The best value of lambda is ",bestLamb)
input,testClass=getMatrix(uniqueVocabWords,classWiseFileList,nameOfClasses,totCount)
weights=[0 for x in range(len(uniqueVocabWords)+1)]
newWeights=gradientAscent(input,testClass,weights,bestLamb)

###########################
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