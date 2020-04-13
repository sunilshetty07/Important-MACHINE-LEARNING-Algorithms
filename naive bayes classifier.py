import csv
import random
random.seed(0)
from pprint import pprint
import math

def loadcsv(filename):
    reader=csv.reader(open(filename,'r'))
    dataset=[]
    for row in reader:
        inlist=[]
        for i in range(len(row)):
            inlist.append(float(row[i]))
        dataset.append(inlist)
    return dataset

def splitDataset(dataset,splitRatio):
    trainSize=int(len(dataset)*splitRatio)
    trainSet=[]
    copy=list(dataset)
    while len(trainSet)<trainSize:
        index=random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]

def sepearateByClass(dataset):
     separated={}
     for i in range(len(dataset)):
        vector=dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
        pprint(separated)
     return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    varience=sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(varience)

def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated=sepearateByClass(dataset)
    summaries={}
    for classValue,instance in separated.items():
        summaries[classValue]=summarize(instance)
    return summaries

def calculateProbability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilites(summaries,inputVector):
    probabilities={}
    for classValue,classSummaries in summaries.items():
        probabilities[classValue]=1
        for i in range(len(classSummaries)):
            mean,stdev=classSummaries[i]
            x=inputVector[i]
            probabilities[classValue]*=calculateProbability(x,mean,stdev)
    return probabilities

def predict(summaries,inputVector):
    probabilities=calculateClassProbabilites(summaries,inputVector)
    bestLabel,bestProb=None,-1
    for classValue,probability in probabilities.items():
        if bestLabel is None or probability>bestProb:
            bestLabel=classValue
            bestProb=probability
    return bestLabel

def getPredictions(summaries,testSet):
    predictions=[]
    for i in range(len(testSet)):
        result=predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet,predictions):
    correct=0
    for i in range(len(testSet)):
        if testSet[i][-1]==predictions[i]:
            correct+=1
    return (correct/float(len(testSet)))*100.0

def main():
    filename="PI_Diabetes.csv"
    dataset=loadcsv(filename)
    print("\n total length:",len(dataset))
    splitRatio=0.67
    print('training and testin\n')
    trainingSet,testSet=splitDataset(dataset,splitRatio)
    print("total rows in training set{0} rows".format(len(trainingSet)))
    print("total rows in testing set{0} rows".format(len(testSet)))
    summaries=summarizeByClass(trainingSet)
    print("\nmodel summaries",summaries)
    predictions=getPredictions(summaries,testSet)
    print("\npredictions:",predictions)
    
    accuracy=getAccuracy(testSet,predictions)
    print('accuarcy is:',accuracy)
main()
     
     
     
        

    




        