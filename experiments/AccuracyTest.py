import numpy as np
from tensorflow import keras
import pandas as pd
from src.data.ConversionRelationships import conversion
from src.data.DataGenerator import Component
from src.classification.FCN import FCN
from src.classification.SVM import SVM
from src.classification.RF import RF

from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.feature_based import Catch22Classifier

from datetime import datetime

def generateTrainingData(num=50):
    xTrain = []
    label = []
    for id in range(6):
        component = Component(id=id)
        for itr in range(num):
            data = component.model.getModelData()
            xTrain.append(data)
            label.append([id])

    xTrain = np.array(xTrain)
    label = np.array(label)
    return xTrain, label


def classifierAll(svc, rf, fcn, cnn, rocket, tde, catch22, data, id):
    # SVM RF FCN Fusion cnn,rocket,tde,catch22 right or not
    cRight = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    # predicted result of SVM RF FCN Fusion cnn,rocket,tde,catch22
    predictIDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    xTest = []
    xTest.append(data)
    xTest = np.array(xTest)

    predictID = cnn.predict(xTest)
    predictIDs[4] = predictID
    if predictID == id:
        cRight[4] = cRight[4] + 1

    predictID = rocket.predict(xTest)
    predictIDs[5] = predictID
    if predictID == id:
        cRight[5] = cRight[5] + 1

    predictID = tde.predict(xTest)
    predictIDs[6] = predictID
    if predictID == id:
        cRight[6] = cRight[6] + 1

    predictID = catch22.predict(xTest)
    predictIDs[7] = predictID
    if predictID == id:
        cRight[7] = cRight[7] + 1

    predictID = svc.predict(xTest)
    weight[predictID] = weight[predictID] + svc.p
    predictIDs[0] = predictID
    if predictID == id:
        cRight[0] = cRight[0] + 1

    predictID = rf.predict(xTest)
    weight[predictID] = weight[predictID] + rf.p
    predictIDs[1] = predictID
    if predictID == id:
        cRight[1] = cRight[1] + 1

    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    predictID = fcn.predict(xTest)
    weight[predictID] = weight[predictID] + fcn.p
    predictIDs[2] = predictID
    if predictID == id:
        cRight[2] = cRight[2] + 1

    predictID = np.argmax(weight)
    predictIDs[3] = predictID
    if predictID == id:
        cRight[3] = cRight[3] + 1

    return predictIDs, cRight


def accuracy(numGenerated, sigma, r, numClassified):
    xTrain, label = generateTrainingData(num=numGenerated)

    print("CNNClassifier training")
    cnn = CNNClassifier()
    cnn.fit(xTrain, np.ravel(label))
    print("RocketClassifier training")
    rocket = RocketClassifier(rocket_transform="multirocket", n_jobs=-1)
    rocket.fit(xTrain, np.ravel(label))
    print("TemporalDictionaryEnsemble training")
    tde = TemporalDictionaryEnsemble(n_jobs=-1)
    tde.fit(xTrain, np.ravel(label))
    print("Catch22Classifier training")
    catch22 = Catch22Classifier(n_jobs=-1)
    catch22.fit(xTrain, np.ravel(label))

    print("SVM training")
    svc = SVM()
    svc.fit(xTrain, np.ravel(label))
    print("RF training")
    rf = RF()
    rf.fit(xTrain, np.ravel(label))
    print("FCN training")
    nb_classes = len(np.unique(label))
    label = keras.utils.to_categorical(label, nb_classes)
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    fcn = FCN(xTrain.shape[1:], nb_classes)
    fcn.fit(xTrain, label)

    id = 0
    number = np.array([1, 0, 0, 0, 0, 0])  # 各类别数量
    classRight = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0])  # SVM RF FCN Fusion cnn,rocket,tde,catch22 Model right or not 各分类方法类别正确数
    transRight = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0])  # SVM RF FCN Fusion cnn,rocket,tde,catch22 Relationship right or not 各分类方法转换正确数

    data = Component(id).getDataWithBothNoise(sigma=sigma, r=r)
    predictIDs, cRight = classifierAll(svc, rf, fcn, cnn, rocket, tde, catch22, data, id)
    classRight = classRight + cRight
    for i in range(numClassified):
        print(i, "    ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        oldID = id
        oldpredictIDs = predictIDs
        id = conversion(id=id)
        data = Component(id).getDataWithBothNoise(sigma=sigma, r=r)

        predictIDs, cRight = classifierAll(svc, rf, fcn, cnn, rocket, tde, catch22, data, id)
        classRight = classRight + cRight
        for j in range(len(classRight)):
            if oldpredictIDs[j] == oldID and predictIDs[j] == id:
                transRight[j] = transRight[j] + 1

        number[id] = number[id] + 1

    keras.backend.clear_session()

    return classRight / numClassified, transRight / (numClassified - 1)


def test(numGenerated=50, sigma=0.1, r=0.1, numClassified=1000):
    df = pd.DataFrame(
        columns=['itr', 'SVM_M', 'RF_M', 'FCN_M', 'Fusion_M', 'cnn_M', 'rocket_M', 'tde_M', 'catch22_M', 'SVM_R',
                 'RF_R', 'FCN_R', 'Fusion_R', 'cnn_R', 'rocket_R', 'tde_R', 'catch22_R'], dtype=object)

    for itr in range(10):
        accM, accR = accuracy(numGenerated, sigma, r, numClassified)
        print(itr, accM, accR, sep="\t")
        df = df.append({'itr': itr, 'SVM_M': accM[0], 'RF_M': accM[1], 'FCN_M': accM[2],
                        'Fusion_M': accM[3], 'cnn_M': accM[4], 'rocket_M': accM[5], 'tde_M': accM[6],
                        'catch22_M': accM[7], 'SVM_R': accR[0], 'RF_R': accR[1], 'FCN_R': accR[2], 'Fusion_R': accR[3],
                        'cnn_R': accR[4], 'rocket_R': accR[5], 'tde_R': accR[6],
                        'catch22_R': accR[7]}, ignore_index=True)
        resultFileName = "..\\results\\Accuracy.csv"
        df.to_csv(resultFileName)


if __name__ == '__main__':
    test()
