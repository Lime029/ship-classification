import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import adjusted_rand_score
from matplotlib import markers,colors
from utils import loadData, plotVesselTracks
from predictVessel import predictWithK, predictWithoutK

def plotVesselTracks3D(features, labels):
    figure = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(features[:, 1], features[:, 2], features[:, 0], c=labels)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Time')

def predictPosition(latitude, longitude, direction, speed, time_diff):
    # Predict latitude and longitude of a ship at specified amount of time after last known position
    # direction = 0 or 3600 means north (+ lat), 900 is east (+ lon), 1800 is south (- lat), 2700 is west (- lon)
    earth_radius = 3440.065 # Earth radius in nautical miles
    angle = math.radians(direction / 10) # Convert angle from degrees to radians
    displacement = ((speed/10) * time_diff) / 3600.0
    new_latitude = latitude + (displacement / earth_radius) * math.cos(angle)
    new_longitude = longitude + (displacement / earth_radius) * math.sin(angle)
    return new_latitude, new_longitude

def predict(testFeatures, testLabels=None, forward=True):
    size = len(testFeatures)
    predictions = np.zeros(size)
    classes = [] # most recent transmission from each ship/class
    actual_classes = []
    class_index = 0 # number of classes, to keep track of next new class index
    # Maybe sort testFeatures by time just to be safe -- not needed if structured like set3NoVID.csv
    for i in range(size):
        entry = testFeatures[i, :]

        best_diff = float('inf')
        best_c = None
        start = 0
        increment = 1
        end = len(classes)
        if forward == False:
            start = len(classes) - 1
            increment = -1
            end = -1
        for c in range(start,end,increment):
            time_diff = abs(entry[0] - classes[c][0])
            if time_diff > 0:
                actual = (entry[1], entry[2])#(classes[c][1], classes[c][2])
                angle = classes[c][4]#entry[4]
                if forward == False:
                    angle = (classes[c][4] + 1800) % 3600#(entry[4] + 1800) % 3600
                #prediction = predictPosition(entry[1], entry[2], angle, entry[3], time_diff)
                prediction = predictPosition(classes[c][1], classes[c][2], angle, classes[c][3], time_diff)
                #error = ((actual[0] - prediction[0])**2 + (actual[1] - prediction[1])**2) + velocityError(classes[c], entry)
                #threshold = 0.001
                error = (actual[0] - prediction[0])**2 + (actual[1] - prediction[1])**2
                threshold = 0.0001
                if error < threshold and error < best_diff: # Not a new ship
                    # Maybe keep track of all less than error threshold and discriminate further based on angle/speed
                    best_diff = error
                    best_c = c
        if best_c == None: # Set as start of new class
            classes.append(entry)
            if testLabels != None:
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
            if testLabels != None:
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
    return predictions

def evaluate(features, labels, predictions, threeD):
    predNumVessels = np.unique(predictions).size
    ari = adjusted_rand_score(labels, predictions)
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: ' + f'{ari}')
    if threeD:
        plotVesselTracks3D(features, predictions)
        plt.title('Vessel tracks by cluster')
        plotVesselTracks3D(features, labels)
        plt.title('Vessel tracks by label')
        plt.show()
    else:
        plotVesselTracks(features[:,[2,1]], predictions)
        plt.title('Vessel tracks by cluster')
        plotVesselTracks(features[:,[2,1]], labels)
        plt.title('Vessel tracks by label')

def views(features, labels, predictions, redun):
    predNumVessels = np.unique(predictions).size

    plt.figure()
    plt.scatter(features[:, 1], features[:, 2], c=predictions)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    plt.figure()
    plt.scatter(features[:, 1], features[:, 0], c=predictions)
    plt.xlabel('Latitude')
    plt.ylabel('Time')

    plt.figure()
    plt.scatter(features[:, 2], features[:, 0], c=predictions)
    plt.xlabel('Longitude')
    plt.ylabel('Time')
    plt.show()

ship_data = loadData('set1.csv') # OBJECT_ID, VID, TIME, LAT, LON, SPEED_OVER_GROUND, COURSE_OVER_GROUND
ship_data_2 = loadData('set2.csv') # OBJECT_ID, VID, TIME, LAT, LON, SPEED_OVER_GROUND, COURSE_OVER_GROUND
ship_data_3 = loadData('set3NoVID.csv')

features = ship_data[:,2:]
labels = ship_data[:,1]
evaluate(features, labels, predict(features, forward=True), True)
