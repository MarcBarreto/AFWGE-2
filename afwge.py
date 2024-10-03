import csv
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, TensorDataset

def generate_population(length, dataset, individual_idx, constraints=[None], population = []):
    """
    Generates a new population for a genetic algorithm based on the dataset's feature ranges, with optional constraints. 
    Each chromosome in the population consists of both feature values and corresponding weights. 

    :param length: The size of the population to be generated.
    :param dataset: The dataset used to determine the range of possible values for each feature.
    :param individual_idx: The index of the individual in the dataset that is used as a reference for constrained features.
    :param constraints: A list of feature names that should not be mutated (i.e., will keep the same value from the original instance).
    :param population: An optional existing population that the new population will be concatenated to.
    :return: A new population where each chromosome consists of feature values and corresponding feature weights.
    """
    new_population = []
    columns = dataset.columns[:-1]
    min_features = dataset.describe().loc['min'][:-1]
    max_features = dataset.describe().loc['max'][:-1]
    
    weight = 1 / len(columns)

    for idx, column in enumerate(columns):
        if pd.api.types.is_numeric_dtype(dataset[column]):
            if column not in constraints:
                if min_features.iloc[idx] == int(min_features.iloc[idx]) and max_features.iloc[idx] == int(max_features.iloc[idx]):
                    new_population.append(np.random.randint(int(min_features.iloc[idx]), int(max_features.iloc[idx]) + 1, length))
                else:
                    new_population.append(np.random.uniform(min_features.iloc[idx], max_features.iloc[idx], length))
            else:
                new_population.append([dataset.iloc[individual_idx].iloc[idx]] * length)
        else:
            if column not in constraints:
                categories = dataset[column].unique()
                new_population.append(np.random.choice(categories, length))
            else:
                category = dataset.iloc[individual_idx].iloc[idx]
                aux = []
                for i in range(length):
                    aux.append(category)
                new_population.append(aux)

    for column in columns:
        new_population.append([weight] * length)

    new_population = np.stack(new_population, axis=-1)

    if len(population) > 0:
        return np.concatenate((population, new_population), axis = 0)
    
    return new_population

def matching_distance(x, c):
    """
    Calculates the weighted distance between two instances, x and c. 
    The distance is computed differently for numerical and categorical features, using feature-specific weights. 

    :param x: The original instance (list or array of feature values).
    :param c: The counterfactual instance (list or array of feature values and corresponding feature weights).
    :return: The weighted distance between the original and counterfactual instances.
    """
    distance = 0
    num_columns = len(x)

    for idx in range(len(x)):
        if isinstance(x[idx], (np.integer, np.floating)):
            distance += abs(x[idx] - c[idx]) * c[idx + num_columns]
        
        else:
            match = 0 if x[idx] == c[idx] else 1
            distance += match * c[idx + num_columns]
            
    return distance

def get_parents_idx(scores, num_parents):
    sorted_indices = np.argsort(scores)
    
    return sorted_indices[:num_parents]

def check_eval_offspring(offspring, parents_idx, population, x, eval_population):
    """
    Evaluates the fitness of an offspring and compares it to the corresponding parents in the population. 
    If the offspring has a better (lower) distance compared to a parent, the parent is replaced by the offspring.

    :param offspring: The new generated offspring (list or array of feature values).
    :param parents_idx: List of indices of the parent individuals in the population.
    :param population: The current population of individuals (list or array).
    :param x: The original instance, used as the reference point for distance evaluation.
    :param eval_population: List or array containing the evaluation scores (distances) of the current population.
    :return: Updated population with potentially replaced individuals.
    """
    for idx in parents_idx:
        offspring_distance = matching_distance(x, offspring)
        if offspring_distance < eval_population[idx]:
            population[idx] = offspring
            eval_population[idx] = offspring_distance
            break

    return population

def crossover(population, x, parents_idx, crossover_prob, eval_population, columns, constraints=[None]):
    """
    Performs crossover on selected parent individuals in the population to generate offspring, 
    considering constraints and evaluation. Swaps features between two parents based on random crossover points, 
    and evaluates the offspring to check if they should replace the parents in the population.

    :param population: The current population of individuals (list or array).
    :param x: The original instance used for reference in matching distance calculation.
    :param parents_idx: List of indices of selected parent individuals for crossover.
    :param crossover_prob: Probability of crossover occurring between two parents.
    :param eval_population: List or array containing the evaluation scores (distances) of the current population.
    :param columns: The list of column names corresponding to features of the dataset.
    :param constraints: List of columns/features that should not be altered during crossover (default: [None]).
    :return: The updated population and the list of generated offsprings.
    """
    offsprings = []
    parents = [population[idx] for idx in parents_idx]
    
    num_features = len(x)
    num_individuals = len(parents)
    
    for i in range(0, num_individuals, 2):
        if np.random.rand() <= crossover_prob and i + 1 < num_individuals:
            parent1 = parents[i].copy()
            parent2 = parents[i+1].copy()
            
            crossover_points = np.random.choice(range(num_features), size=2, replace=False)
            
            for point in crossover_points:
                if columns[point] not in constraints:
                    parent1[point], parent2[point] = parent2[point], parent1[point]
                    parent1[point + num_features], parent2[point + num_features] = parent2[point + num_features], parent1[point + num_features]
            
            population = check_eval_offspring(parent1, [i, i + 1], population, x, eval_population)
            population = check_eval_offspring(parent2, [i, i + 1], population, x, eval_population)
            offsprings.append(parent1)
            offsprings.append(parent2)
        else:
            offsprings.append(population[i])
            if i + 1 < num_individuals:
                offsprings.append(population[i+1])
                
    return population, offsprings

def mutate(population, x, parents_idx, offsprings, mutation_prob, eval_population, columns, constraints):
    """
    Applies mutation to offspring individuals in the population, modifying feature values and their corresponding weights 
    with a given mutation probability. Ensures mutated offspring replace parents in the population if they are a better match 
    based on the evaluation score.

    :param population: The current population of individuals (list or array).
    :param x: The original instance used for reference in the matching distance calculation.
    :param parents_idx: Indices of selected parent individuals that might be replaced by the mutated offspring.
    :param offsprings: List of offspring individuals generated by crossover.
    :param mutation_prob: Probability of mutation occurring on an offspring's features or weights.
    :param eval_population: List or array containing the evaluation scores (distances) of the current population.
    :param columns: List of column names corresponding to features of the dataset.
    :param constraints: List of columns/features that should not be altered during mutation.
    :return: The updated population after mutation.
    """
    num_offsprings = len(offsprings)
    num_features = len(x)
    num_individuals = len(population)
    
    weight = 1 / (2 * num_features)
    
    for i in range(num_offsprings):
        if np.random.rand() <= mutation_prob:
            mutation_points = np.random.choice(range(num_features), size=2, replace=False)
            
            for point in mutation_points:
                if columns[point] not in constraints:
                    if isinstance(offsprings[i][point], (np.integer, np.floating)):
                        idx1, idx2 = np.random.choice(range(num_individuals), size=2, replace=False)
                        offsprings[i][point] = np.mean([population[idx1][point], population[idx2][point]])
                    else:
                        random_idx = np.random.randint(num_individuals)
                        offsprings[i][point] = population[random_idx][point]
                    
            population = check_eval_offspring(offsprings[i],parents_idx, population, x, eval_population)
        
        if np.random.rand() <= mutation_prob:
            weight_points = np.random.choice(range(num_features), size=2, replace=False)

            change_amount = np.random.uniform(0, weight)

            offsprings[i][weight_points[0] + num_features] += change_amount
            offsprings[i][weight_points[1] + num_features] -= change_amount

            population = check_eval_offspring(offsprings[i],parents_idx, population, x, eval_population)
        
    return population

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def fit(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        total = 0
        accuracy = .0
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            
            optimizer.zero_grad()
            
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            _, predicted = torch.max(outputs.data, axis = 1)
            total += labels.size(0)

            accuracy += (predicted == labels).sum().item()

            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f} Accuracy: {accuracy/total:.2f}')

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

def inference(model, data):
    model.eval()

    with torch.no_grad():
        if len(data.shape) == 1:
            data.squeeze(0)
        output = model(data)
        _, result = torch.max(output.data, 1)
        
        return result

def train_model(X, y, train_loader, test_loader, test = False):
    input_size = X.shape[1]
    output_size = len(set(y))
    learning_rate = 0.001
    epochs = 30

    model = MLP(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fit(model, train_loader, criterion, optimizer, epochs)
    
    if test:
        test_model(model, test_loader)
    
    return model

def iris_preprocess(iris):
    """
    Preprocesses the Iris dataset by splitting the data into features and target, scaling the features, and 
    converting the data into PyTorch tensors. Additionally, it creates train and test data loaders for model training.

    :param iris: The Iris dataset as a dictionary containing 'data' (features) and 'target' (labels).
    :return: A tuple containing:
        - scaler: The fitted StandardScaler used to scale the features.
        - X: The scaled feature matrix as a NumPy array.
        - y: The target labels as a NumPy array.
        - iris_df: A DataFrame representation of the Iris dataset.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
    """
    sepal_length, sepal_width, petal_length, petal_width = zip(*iris['data'])

    iris_dict = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width,
            'target': iris['target'] }

    iris_df = pd.DataFrame(iris_dict)

    X = iris_df.drop('target', axis=1).values
    y = iris_df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return scaler, X, y, iris_df, train_loader, test_loader

def filter_population(scaler, mutated_population, x, model):
    """
    Filters out individuals from the mutated population that match the target class of the input instance 'x'.
    
    The function transforms the features of the mutated population using the given scaler, makes predictions 
    using the provided model, and removes individuals whose predicted class matches the target class of 'x'.

    :param scaler: A fitted StandardScaler used to normalize the features of the mutated population.
    :param mutated_population: A NumPy array containing the mutated population with features and weights.
    :param x: The original instance (dictionary-like) containing the target class for comparison.
    :param model: The machine learning model used for prediction.
    :return: A filtered NumPy array containing individuals whose predicted class does not match the target class of 'x'.
    """
    filtered_population = []

    results = inference(model, torch.tensor(scaler.transform(mutated_population[:,:len(x) - 1]), dtype = torch.float32))
                       
    for idx, result in enumerate(results):
        if result != x['target']:
            filtered_population.append(mutated_population[idx])
            
    return np.array(filtered_population)

def contrafactual_afwge(scaler, model, dataset, select=2, k=100, generations=10, constraints=[None], q=0.7, pc=0.7, pm=0.2):
    """
    Generates contrafactual examples using an adaptive feature weight genetic algorithm (AFWGE).

    The function iterates through each instance in the dataset, creating a population of potential contrafactuals 
    and evolving them through selection, crossover, mutation, and filtering based on a given model's predictions. 
    It returns a list of contrafactual examples that are closest to the original instance's feature values 
    but have different target predictions.

    :param scaler: A fitted StandardScaler used to normalize the features of the dataset.
    :param model: The machine learning model used for prediction to evaluate the contrafactuals.
    :param dataset: The input dataset containing features and target classes.
    :param select: The number of parents selected for crossover.
    :param k: The size of the population to generate.
    :param generations: The maximum number of generations for the genetic algorithm.
    :param constraints: A list of features that are constrained and should not be mutated.
    :param q: The fraction of the population to select for contrafactual generation.
    :param pc: The probability of crossover between pairs of parents.
    :param pm: The probability of mutation for individuals in the population.
    :return: A list of tuples containing the original instance and its corresponding contrafactual example.
    """
    columns = dataset.columns[:-1]
    
    contrafactuals = []
    
    for idx in range(dataset.shape[0]):
        x = dataset.iloc[idx].values[:-1]

        population = []
        
        terminationCounter = 0
        best_fitness = float('inf')
        
        for i in range(generations):
            population = generate_population(k - len(population), dataset, idx, constraints, population)
            
            eval_population = np.array([matching_distance(x, individual) for individual in population])

            parents_idx = get_parents_idx(eval_population, select)

            population, offsprings = crossover(population, x, parents_idx, pc, eval_population, columns, constraints)

            population = mutate(population, x, parents_idx, offsprings, pm, eval_population, columns, constraints)

            population = filter_population(scaler, population, dataset.iloc[idx], model)
            
            eval_population = np.array([matching_distance(x, individual) for individual in population])

            idx_sorted = sorted(range(len(eval_population)), key=lambda i: eval_population[i])

            if eval_population[idx_sorted[0]] < best_fitness:
                best_fitness = eval_population[idx_sorted[0]]
                
                terminationCounter = 0
            else:
                terminationCounter += 1
            
            if terminationCounter >= 5:
                break

            if len(population) == k and i != generations - 1:
                j = random.randint(-k // 2, k - 2)
                
                population = population[idx_sorted[:j]]

                eval_population = eval_population[idx_sorted[:j]]

                idx_sorted = sorted(range(len(eval_population)), key=lambda i: eval_population[i])
        
        length = int(q * len(idx_sorted))
        for i in range(length):
            contrafactuals.append((x, population[idx_sorted[i]]))
    
    return contrafactuals

def export_contrafactuals_custom_to_csv(contrafactuals, iris_df, filename='contrafactuals.csv'):
    """
    Exports contrafactual examples to a CSV file in a custom format.

    The CSV file will contain rows of contrafactuals, where each row includes:
    - The original feature values from the input dataset.
    - The contrafactual feature values generated by the algorithm.
    - The corresponding weights assigned to the contrafactual features.

    The headers of the CSV file are formatted to clearly distinguish between original 
    values, contrafactual values, and weights.

    :param contrafactuals: A list of tuples, where each tuple contains the original instance 
                          and its corresponding contrafactual example.
    :param iris_df: The original DataFrame from which the feature names are extracted.
    :param filename: The name of the CSV file to save the contrafactual examples. 
                     Defaults to 'contrafactuals.csv'.
    """
    original_headers = [f'{name}_original' for name in iris_df.columns[:-1]]
    contrafactual_headers = [f'{name}_contrafactual' for name in iris_df.columns[:-1]]
    weight_headers = [f'{name}_weight' for name in iris_df.columns[:-1]]
    
    headers = original_headers + contrafactual_headers + weight_headers

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(headers)
        
        for i in range(len(contrafactuals)):
            row =  list(contrafactuals[i][0]) + list(np.round(contrafactuals[i][1], 4))
            writer.writerow(row)
    
    print(f'{filename} saved!')

def main():
    iris = datasets.load_iris()

    scaler, X, y, iris_df, train_loader, test_loader = iris_preprocess(iris)

    model = train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Contrafactuals ----------------')

    contrafactuals = contrafactual_afwge(scaler, model = model, dataset = iris_df, k = 1000, constraints = [], select = 20)

    export_contrafactuals_custom_to_csv(contrafactuals, iris_df)

    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()