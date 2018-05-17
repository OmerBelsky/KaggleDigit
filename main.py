import tensorflow as tf
import numpy as np

def loadData():
    trainSet = []
    trainTargets = []
    testSet = []
    indexes = dict()
    values = dict()
    usable = []
    with open("train.csv", 'r') as reader:
        next(reader)
        for line in reader:
            lineList = list(map(lambda x: int(x), line.split(',')))
            for i in range(len(lineList)):
                if i not in values.keys():
                    values[i] = lineList[i]
                    indexes[i] = False
                elif lineList[i] != values[i]:
                    indexes[i] = True
    with open("test.csv", 'r') as reader:
        next(reader)
        for line in reader:
            lineList = list(map(lambda x: int(x), line.split(',')))
            for i in range(len(lineList)):
                if i not in values.keys():
                    values[i] = lineList[i]
                    indexes[i] = False
                elif lineList[i] != values[i]:
                    indexes[i] = True
    for key in indexes.keys():
        if indexes[key]:
            usable.append(key)
    with open("train.csv", 'r') as reader:
        next(reader)
        for line in reader:
            lineList = list(map(lambda x: int(x), line.split(',')))
            target, pixels = lineList[0], lineList[1:]
            pixels = [pixels[i]/255.0 for i in usable]
            trainSet.append(pixels)
            trainTargets.append(target)
    with open("test.csv", 'r') as reader:
        next(reader)
        for line in reader:
            pixels = line.strip().split(',')
            pixels = [pixels[i] for i in usable]
            testSet.append(list(map(lambda x: int(x)/255.0, pixels)))
    return trainSet, trainTargets, testSet

def predictWithDNN():
    featureColumns = [tf.contrib.layers.real_valued_column("", dimension=728)]
    trainFile = "DNNTrain.csv"
    testFile = "DNNTest.csv"
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=featureColumns, hidden_units=[10, 20, 10],
                                                n_classes=10)
    trainData = tf.contrib.learn.datasets.base.load_csv_with_header(filename=trainFile, target_dtype=np.int,
                                                                    features_dtype=np.float)
    testData = tf.contrib.learn.datasets.base.load_csv_with_header(filename=testFile, target_dtype=np.int,
                                                                   features_dtype=np.float)
    classifier.fit(input_fn=lambda: (tf.constant(trainData.data), tf.constant(trainData.target)), steps=4000)
    predictions = classifier.predict_classes(
        input_fn=lambda: (tf.constant(testData.data), tf.constant(testData.target)))
    with open("dnn5.csv", 'a') as writer:
        writer.write("ImageId,Label")
        i = 1
        for prediction in predictions:
            writer.write('\n')
            writer.write("{},{}".format(i, prediction))
            i += 1

"""trainSet, trainTargets, testSet = loadData()
with open("DNNTest.csv", 'a') as writer:
    writer.write("{},{},0,1,2,3,4,5,6,7,8,9".format(len(testSet), len(testSet[0])))
    for i in range(len(testSet)):
        writer.write('\n')
        for elem in testSet[i]:
            writer.write('{},'.format(elem))
        writer.write('0')
with open("DNNTrain.csv", 'a') as writer:
    writer.write("{},{},0,1,2,3,4,5,6,7,8,9".format(len(trainSet), len(trainSet[0])))
    for i in range(len(trainSet)):
        writer.write('\n')
        for elem in trainSet[i]:
            writer.write('{},'.format(elem))
        writer.write(str(trainTargets[i]))"""
predictWithDNN()
