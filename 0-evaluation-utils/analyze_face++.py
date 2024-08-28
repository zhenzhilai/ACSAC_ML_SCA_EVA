### import packages for reading csvs and ploting graphs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

#######################################################
################ Data here ###########################
#######################################################



folder = "./"
file_name = sys.argv[1]
if_MANIFOLD = False

def compute_accuracy(folder, file_name, MANIFOLD=False):
    df = pd.read_csv(folder+file_name, sep=' ')
    names = df['name'].tolist()
    confidences = df['confidence'].tolist()
    thresholds_zero = df['threshold0'].tolist()
    thresholds_one = df['threshold1'].tolist()
    thresholds_two = df['threshold2'].tolist()
    size_of_data = len(names)

    ### We need to remove -2 from the confidences
    non_face_count = confidences.count(-2)
    confidences = [c for c in confidences if c != -2]
    thresholds_zero = [t for t in thresholds_zero if t != -99]
    thresholds_one = [t for t in thresholds_one if t != -99]
    thresholds_two = [t for t in thresholds_two if t != -99]

    ### Compute Manifold accuracy    
    manifold_accuracy = len([c for c in confidences if c > 50]) / len(confidences)

    ### if confidence > threshold0, they are from the same person
    same_thresholds_zero = [1 if c > t else 0 for c, t in zip(confidences, thresholds_zero)]
    same_thresholds_one = [1 if c > t else 0 for c, t in zip(confidences, thresholds_one)]
    same_thresholds_two = [1 if c > t else 0 for c, t in zip(confidences, thresholds_two)]

    ### Compute and print the accuracy
    accuracy_zero = sum(same_thresholds_zero)/size_of_data
    accuracy_one = sum(same_thresholds_one)/size_of_data
    accuracy_two = sum(same_thresholds_two)/size_of_data
    non_face = non_face_count/size_of_data

    return accuracy_zero, accuracy_one, accuracy_two, non_face, manifold_accuracy

### invoke function
accuracy_zero, accuracy_one, accuracy_two, non_face, manifold_accuracy = compute_accuracy(folder, file_name, if_MANIFOLD)
print("Accuracy for threshold 0: ", accuracy_zero)
print("Accuracy for threshold 1: ", accuracy_one)
print("Accuracy for threshold 2: ", accuracy_two)
print("Non face count: ", non_face)
print("Manifold accuracy: ", manifold_accuracy)
print(accuracy_zero, accuracy_one, accuracy_two, non_face, round(manifold_accuracy,3))

