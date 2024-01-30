# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def predictPosition(latitude, longitude, direction, speed, time_diff):
    # Predict latitude and longitude of a ship at specified amount of time after last known position
    # direction = 0 or 3600 means north (+ lat), 900 is east (+ lon), 1800 is south (- lat), 2700 is west (- lon)
    earth_radius = 3440.065 # Earth radius in nautical miles
    angle = math.radians(direction / 10) # Convert angle from degrees to radians
    displacement = ((speed/10) * time_diff) / 3600.0
    new_latitude = latitude + (displacement / earth_radius) * math.cos(angle)
    new_longitude = longitude + (displacement / earth_radius) * math.sin(angle)
    return new_latitude, new_longitude

def predict(testFeatures, k=0, testLabels=[], forward=True, eligibles=[]):
    size = len(testFeatures)
    predictions = np.zeros(size)
    classes = [] # most recent transmission from each ship/class
    actual_classes = []
    class_index = 0 # number of classes, to keep track of next new class index
    # Maybe sort testFeatures by time just to be safe -- not needed if structured like set3NoVID.csv
    start = 0
    increment = 1
    end = size
    if forward == False:
        start = size - 1
        increment = -1
        end = -1
    for i in range(start, end, increment):
        entry = testFeatures[i, :]
        best_diff = float('inf')
        best_c = None
        num_eligible = 0
        for c in range(len(classes)):
            time_diff = abs(entry[0] - classes[c][0])
            if time_diff > 0:
                actual = (entry[1], entry[2])
                angle = classes[c][4]
                if forward == False:
                    angle = (classes[c][4] + 1800) % 3600
                prediction = predictPosition(classes[c][1], classes[c][2], angle, classes[c][3], time_diff)
                error = (actual[0] - prediction[0])**2 + (actual[1] - prediction[1])**2
                threshold = 0.0001
                if error < threshold and error < best_diff: # Not a new ship
                    # Maybe keep track of all less than error threshold and discriminate further based on angle/speed
                    best_diff = error
                    best_c = c
                    num_eligible += 1
        eligibles.append(num_eligible)
        if best_c == None: # Set as start of new class
            classes.append(entry)
            if len(testLabels) > 0:
                if testLabels[i] in actual_classes:
                    # Incorrectly adding a new class
                    print('Incorrectly adding new class')
                    print('New entry:', entry)
                    print('Last entry:', classes[actual_classes.index(testLabels[i])])
                    print()
                actual_classes.append(testLabels[i])
            best_c = class_index
            class_index += 1
        else:
            if len(testLabels) > 0:
                if testLabels[i] not in actual_classes:
                    # Should be adding a new class
                    print('Incorrectly failing to add new class')
                    print('New entry:', entry)
                    print('Connected entry:', classes[best_c])
                    print()
                elif actual_classes.index(testLabels[i]) != best_c:
                    # Adding to wrong class
                    print('Adding to incorrect class')
                    print('New entry:', entry)
                    print('Connected entry:', classes[best_c])
                    print('Last entry:', classes[actual_classes.index(testLabels[i])])
                    print()
            classes[best_c] = entry
        predictions[i] = best_c
    # Deal with case when there's too many clusters (reassign smallest ones)
    classes, counts = np.unique(predictions, return_counts=True)
    if k > 0 and classes.size > k:
        remove_classes = np.argsort(counts)[:(classes.size - k)]
        classes_recent = [np.zeros(5) for i in range(len(classes))]
        start = 0
        increment = 1
        end = size
        if forward == False:
            start = size - 1
            increment = -1
            end = -1
        for i in range(start, end, increment):
            entry = testFeatures[i, :]
            if predictions[i] in remove_classes:
                best_diff = float('inf')
                best_c = None
                for c in range(len(classes_recent)):
                    if classes_recent[c][0] == 0:
                        continue
                    time_diff = abs(entry[0] - classes_recent[c][0])
                    if time_diff > 0:
                        actual = (entry[1], entry[2])
                        angle = classes_recent[c][4]
                        if forward == False:
                            angle = (classes_recent[c][4] + 1800) % 3600
                        prediction = predictPosition(classes_recent[c][1], classes_recent[c][2], angle,
                                                     classes_recent[c][3], time_diff)
                        error = (actual[0] - prediction[0])**2 + (actual[1] - prediction[1])**2
                        if error < best_diff:
                            best_diff = error
                            best_c = c
                predictions[i] = best_c
            else:
                classes_recent[int(predictions[i])] = entry
    return predictions

def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    return predict(testFeatures, forward=True, k=numVessels)

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predict(testFeatures, forward=True)
