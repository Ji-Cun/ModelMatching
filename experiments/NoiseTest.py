from experiments.AccuracyTest import generateTrainingData
import pandas as pd
import numpy as np
from tensorflow import keras
from src.classification.FCN import FCN
from src.classification.SVM import SVM
from src.classification.RF import RF
from src.data.ConversionRelationships import conversion
from src.data.DataGenerator import Component

def classifierResult(svc, rf, fcn,data):
    weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xTest = []
    xTest.append(data)
    xTest = np.array(xTest)

    predictID = svc.predict(xTest)
    weight[predictID] = weight[predictID]+svc.p

    predictID = rf.predict(xTest)
    weight[predictID] = weight[predictID] + rf.p

    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    predictID = fcn.predict(xTest)
    weight[predictID] = weight[predictID] + fcn.p

    predictID=np.argmax(weight)

    return predictID

def accuracy(numGenerated,sigma,r,numClassified):
    xTrain, label = generateTrainingData(num=numGenerated)

    svc = SVM()
    svc.fit(xTrain, np.ravel(label))
    rf = RF()
    rf.fit(xTrain, np.ravel(label))
    nb_classes = len(np.unique(label))
    label = keras.utils.to_categorical(label, nb_classes)
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    fcn = FCN(xTrain.shape[1:], nb_classes)
    fcn.fit(xTrain, label)

    id = 0
    number = np.array([1, 0, 0, 0, 0, 0])  # 各类别数量
    classRight=0 #类标签正确数
    transRight=0 #转换正确数

    data = Component(id).getDataWithBothNoise(sigma=sigma, r=r)
    predictID=classifierResult(svc,rf,fcn, data)
    if predictID==id:
        classRight = classRight + 1

    for i in range(numClassified-1):
        oldID = id
        oldpredictID=predictID
        id = conversion(id=id)
        data = Component(id).getDataWithBothNoise(sigma=sigma, r=r)
        predictID=classifierResult(svc,rf,fcn, data)
        if predictID == id:
            classRight = classRight + 1
            if oldpredictID==oldID:
                transRight=transRight+1

        number[id] = number[id] + 1
    keras.backend.clear_session()

    return classRight/numClassified, transRight/(numClassified-1)


def testR(numGenerated=50,sigma=0,numClassified=1000):
    df = pd.DataFrame(columns=['itr', 'r', 'Fusion_M','Fusion_R'], dtype=object)

    for itr in range(10):
        for i in range(11):
            r=i*0.1
            accM, accR = accuracy(numGenerated, sigma, r, numClassified)
            print(itr,r,accM,accR,sep="\t")
            df = df.append({'itr': itr, 'r':r, 'Fusion_M':accM,'Fusion_R':accR}, ignore_index=True)
            resultFileName = "..\\results\\TestRAccuracy.csv"
            df.to_csv(resultFileName)

def testSigma(numGenerated=50,r=0,numClassified=1000):
    df = pd.DataFrame(columns=['itr',  'sigma','Fusion_M','Fusion_R'], dtype=object)

    for itr in range(10):
        for j in range(11):
            sigma = j * 0.1

            accM, accR = accuracy(numGenerated, sigma, r, numClassified)
            print(itr,sigma,accM,accR,sep="\t")
            df = df.append({'itr': itr,  'sigma':sigma,'Fusion_M':accM,'Fusion_R':accR}, ignore_index=True)
            resultFileName = "..\\results\\TestSignaAccuracy.csv"
            df.to_csv(resultFileName)

if __name__ == '__main__':
    testR(numGenerated=50, numClassified=1000)
    testSigma(numGenerated=50, numClassified=1000)
