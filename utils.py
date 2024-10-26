import csv
import numpy as np
import matplotlib.pyplot as plt
from afwge import AFWGE

def export_counterfactuals_custom_to_csv(counterfactuals, iris_df, filename='counterfactuals.csv'):
    """
    Exports counterfactual examples to a CSV file in a custom format.

    The CSV file will contain rows of counterfactuals, where each row includes:
    - The original feature values from the input dataset.
    - The counterfactual feature values generated by the algorithm.
    - The corresponding weights assigned to the counterfactual features.

    The headers of the CSV file are formatted to clearly distinguish between original 
    values, counterfactual values, and weights.

    :param counterfactuals: A list of tuples, where each tuple contains the original instance 
                          and its corresponding counterfactual example.
    :param iris_df: The original DataFrame from which the feature names are extracted.
    :param filename: The name of the CSV file to save the counterfactual examples. 
                     Defaults to 'counterfactuals.csv'.
    """
    original_headers = [f'{name}_original' for name in iris_df.columns[:-1]]
    contrafactual_headers = [f'{name}_counterfactual' for name in iris_df.columns[:-1]]
    weight_headers = [f'{name}_weight' for name in iris_df.columns[:-1]]
    
    headers = original_headers + contrafactual_headers + weight_headers

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(headers)
        
        for i in range(len(counterfactuals)):
            row =  list(counterfactuals[i][0]) + list(np.round(counterfactuals[i][1], 4))
            writer.writerow(row)
    
    print(f'{filename} saved!')

def calc_changed_features(original, counterfactual):
    """
    Calculate the number of features that have changed between the original and counterfactual values.
    
    This function counts the number of features that differ between the original
    feature values and the counterfactual feature values.
    
    :param original: A list or array of original feature values.
    :param counterfactual: A list or array of counterfactual feature values.
    :return: An integer representing the number of features that have changed.
    """
    counter = 0
    for idx in range(len(original)):
        if original[idx] != counterfactual[idx]:
            counter += 1 
    
    return counter

def calculate_metrics(counterfactuals):
    """
    Calculate various metrics for a list of counterfactuals.
    
    This function computes the total and average weighted distance, as well as the
    number of changed features between the original values and the counterfactual
    values for a given set of counterfactuals. Each counterfactual contains original
    values and modified values with their corresponding feature weights.
    
    :param counterfactuals: A list of tuples, where each tuple contains:
        - original: A list or array of original feature values.
        - modified_with_weights: A list or array of modified feature values and their weights.
    :return: A tuple containing:
        - average_weighted_distance: The average weighted distance across all counterfactuals.
        - distances: A list of individual weighted distances for each counterfactual.
        - changed_features: The total number of changed features across all counterfactuals.
    """
    total_weighted_distance = 0.0
    distances = []
    num_counterfactuals = len(counterfactuals)
    
    changed_features = 0
    
    for original, modified_with_weights in counterfactuals:
        distance = AFWGE.matching_distance(original, modified_with_weights)
        total_weighted_distance += distance
        distances.append(distance)
        
        changed_features += calc_changed_features(original, modified_with_weights[:len(original)])
    
    average_weighted_distance = total_weighted_distance / num_counterfactuals
    
    return average_weighted_distance, distances, changed_features

def box_plot(x, title):
    """
    Create and display a box plot for a given set of values.
    
    This function generates a box plot for the provided data, customizing
    the appearance of the plot, including colors, outliers, and grid lines.
    
    :param x: A list or array of features distances to be plotted.
    :param title: A string representing the title of the plot.
    """
    plt.figure(figsize=(8, 5))

    box = plt.boxplot(x, patch_artist=True, showfliers=True, notch=True)

    box['boxes'][0].set_facecolor('blue')
    box['boxes'][0].set_edgecolor('black')
    box['boxes'][0].set_linewidth(1)

    for flier in box['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)

    plt.title(title, fontsize=16)
    plt.ylabel('Distance', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks([]) 
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()
