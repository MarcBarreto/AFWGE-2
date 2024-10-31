import torch
import random
import numpy as np
import pandas as pd
from mlp import inference
from utils import generate_random_value, matching_distance

class AFWGE:
    def __init__(self, model, scaler, dataset, constraints=[None], partial_constraints = {}, select=2, k=100, generations=10, q=0.7, pc=0.7, pm=0.2, encoded = False, encoded_columns = []):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.columns = dataset.columns[:-1]
        self.constraints = constraints
        self.partial_constraints = partial_constraints
        self.select = select
        self.k = k
        self.generations = generations
        self.q = q
        self.pc = pc
        self.pm = pm
        self.encoded = encoded
        self.encoded_columns = encoded_columns

    def generate_population(self, length, individual_idx, population=[], eval_population = []):
        """
        Generates a new population of individuals, where each individual is composed of feature values and corresponding weights.
        The population is generated based on the value ranges of the dataset's features.

        :param length: The size of the population to be generated.
        :param individual_idx: The index of the reference individual in the dataset, used to apply constraints.
        :param population: Existing population, if any, to which the new population will be concatenated.
        :return: A new population where each individual contains feature values and their corresponding weights.
        """
        columns = self.dataset.columns[:-1]
        min_features = self.dataset.describe().loc['min'][:-1]
        max_features = self.dataset.describe().loc['max'][:-1]
        weight = 1 / len(columns)
        new_population = []

        numb_idx = 0
    
        for idx, column in enumerate(columns):
            if pd.api.types.is_numeric_dtype(self.dataset[column]):
                if column not in self.constraints:
                    if self.partial_constraints and column in self.partial_constraints.keys():
                        constraint = partial_constraints[column]
                        current_value = dataset.iloc[individual_idx][column]
                        if constraint == 'up':
                            if min_features.iloc[numb_idx] == int(min_features.iloc[numb_idx]) and max_features.iloc[numb_idx] == int(max_features.iloc[numb_idx]):
                                new_population.append(np.random.randint(int(current_value), int(max_features.iloc[numb_idx]) + 1, length))
                            else:
                                new_population.append(np.random.uniform(current_value, max_features.iloc[numb_idx], length))
                        elif constraint == 'down':
                            if min_features.iloc[numb_idx] == int(min_features.iloc[numb_idx]) and max_features.iloc[numb_idx] == int(max_features.iloc[numb_idx]):
                                new_population.append(np.random.randint(int(min_features.iloc[numb_idx]), int(current_value), length))
                            else:
                                new_population.append(np.random.uniform(min_features.iloc[numb_idx], current_value, length))
                    else:
                        if min_features.iloc[numb_idx] == int(min_features.iloc[numb_idx]) and max_features.iloc[numb_idx] == int(max_features.iloc[numb_idx]):
                            new_population.append(np.random.randint(int(min_features.iloc[numb_idx]), int(max_features.iloc[numb_idx]) + 1, length))
                        else:
                            new_population.append(np.random.uniform(min_features.iloc[numb_idx], max_features.iloc[numb_idx], length))
                else:
                    new_population.append([self.dataset.iloc[individual_idx].iloc[numb_idx]] * length)
                numb_idx += 1
            else:
                if column not in self.constraints:
                    categories = self.dataset[column].unique()
                    new_population.append(np.random.choice(categories, length))
                else:
                    category = self.dataset.iloc[individual_idx].iloc[idx]
                    new_population.append([category] * length)

        for _ in range(len(columns)):
            new_population.append([weight] * length)

        new_population = np.stack(new_population, axis=-1)
        
        new_eval_population = np.array([matching_distance(self.dataset.iloc[idx].values[:-1], individual) for individual in new_population])
        
        if len(population) > 0:
            return np.concatenate((population, new_population), axis=0), np.concatenate((eval_population, new_eval_population), axis=0)

        return new_population, new_eval_population

    def get_parents_idx(self, scores):
        """
        Selects the best individuals from the population based on their evaluation scores.
        The number of selected parents is defined by the parameter 'select'.

        :param scores: Array of evaluation scores for the current population.
        :return: Indices of the top-scoring individuals to be selected as parents.
        """
        sorted_indices = np.argsort(scores)
        return sorted_indices[:self.select]

    def check_eval_offspring(self, offsprings, parents_idx, population, x, eval_population):
        """
        Evaluates the fitness of offsprings by calculating them distance to the original instance 'x'. 
        If the offspring has better fitness (lower distance) than a parent, the parent is replaced by the offspring.

        :param offsprings: The new generated offsprings.
        :param parents_idx: Indices of the parent individuals in the population.
        :param population: Current population of individuals.
        :param x: The original instance.
        :param eval_population: Evaluation scores of the current population.
        Returns:
        - Updated population where some individuals may be replaced by offspring.
        - Updated evaluation scores for the population.
        """
        for idx in parents_idx:
            for offspring in offsprings:
                offspring_distance = matching_distance(x, offspring)
                if offspring_distance < eval_population[idx]:
                    population[idx] = offspring
                    eval_population[idx] = offspring_distance
                    break

        return population, eval_population

    def elitist_check_eval_offspring(self, offsprings, parents_idx, population, x, eval_population):
        """
        Evaluate and selectively retain offspring using an elitist strategy.
        
        This function compares each offspring's evaluation score with those of the weakest individuals 
        in the current population, replacing individuals if the offspring perform better. The function 
        ensures that only the best individuals are kept in the population, enhancing the selection pressure 
        towards optimal solutions. If some offspring are not selected, the function chooses additional 
        parents as placeholders based on evaluation scores.

        Parameters:
        - offsprings: List of newly generated offspring individuals to evaluate.
        - parents_idx: List of indices for the parent individuals in the population.
        - population: Current population of individuals.
        - x: The original instance, used for calculating the distance to each offspring.
        - eval_population: Array of evaluation scores for the current population.
        
        Returns:
        - Updated population where some individuals may be replaced by offspring.
        - Updated evaluation scores for the population.
        - List of indices of the offspring that were successfully added to the population.
        """
        idx_sorted = np.argsort(eval_population)
        eval_idx = len(eval_population) - 1
        offsprings_idx = []

        for offspring in offsprings:
            offspring_score = matching_distance(x, offspring)
            individual_idx = idx_sorted[eval_idx]
            
            if offspring_score < eval_population[individual_idx]:
                population[individual_idx] = offspring
                eval_population[individual_idx] = offspring_score
                offsprings_idx.append(individual_idx)
                eval_idx -= 1

        missing_offsprings = len(offsprings) - len(offsprings_idx)
        if missing_offsprings > 0:
            sorted_parents = sorted(parents_idx, key=lambda i: eval_population[i])
            offsprings_idx.extend(sorted_parents[:missing_offsprings])

        return population, eval_population, offsprings_idx

    def crossover(self, population, x, parents_idx, eval_population, pc, elitist):
        """
        Performs crossover between selected parent individuals to generate offspring.
        
        This method takes a subset of the current population, identified by `parents_idx`, and applies 
        crossover operations to generate new offspring. During crossover, feature values are swapped 
        between pairs of parents at randomly selected points, with some attributes constrained based 
        on specific dataset constraints. The newly generated offspring are evaluated and may replace 
        existing individuals based on their evaluation scores. If `elitist` is True, a stricter evaluation 
        is applied to retain only the best-performing individuals.
        
        Parameters:
        - population: The current population of individuals represented as a list or array.
        - x: The original instance being used for counterfactual generation.
        - parents_idx: List of indices representing selected parents in the population for crossover.
        - eval_population: Array of evaluation scores for each individual in the current population.
        - pc: Probability of crossover occurring between a given pair of parents.
        - elitist: Boolean indicating whether to apply elitist selection, which prioritizes offspring with better scores.
        
        Returns:
        - Updated population with potentially new offspring replacing existing individuals.
        - Updated evaluation scores for the population.
        - Indices of the individuals that participated in crossover and were added to the population.
        """
        parents = [population[idx] for idx in parents_idx]
        num_features = len(x)
        num_individuals = len(parents)

        for i in range(0, num_individuals, 2):
            if np.random.rand() <= pc and i + 1 < num_individuals:
                parent1 = parents[i].copy()
                parent2 = parents[i + 1].copy()
                
                crossover_points = np.random.choice(range(num_features), size=2, replace=False)

                for point in crossover_points:
                    if self.dataset.columns[point] not in self.constraints:
                        parent1[point], parent2[point] = parent2[point], parent1[point]
                        parent1[point + num_features], parent2[point + num_features] = parent2[point + num_features], parent1[point + num_features]

                if elitist:
                    population, eval_population, offsprings_idx = self.elitist_check_eval_offspring(
                        [parent1, parent2], [i, i + 1], population, x, eval_population
                    )
                else:
                    population, eval_population = self.check_eval_offspring(
                        [parent1, parent2], [i, i + 1], population, x, eval_population
                    )
                    offsprings_idx = parents_idx

        return population, eval_population, offsprings_idx

    def interpolates_crossover(self, population, x, parents_idx, eval_population, pc):
        """
        Perform a interpolates crossover operation on selected parent pairs, creating offspring by swapping feature values
        at randomly chosen crossover points. If an offspring improves upon the parent, it replaces the parent in the population.

        Parameters:
        - population (list): Current population of individuals, where each individual is represented as a list of feature values.
        - x (list): The original instance against which counterfactuals are generated.
        - parents_idx (list): Indices of selected parent individuals for potential crossover.
        - eval_population (np.ndarray): Evaluation scores for the current population.
        - pc (float): Crossover probability, determining the likelihood of crossover operation for each parent pair.

        Returns:
        - tuple:
        - Updated population with new offspring replacing parents if they yield better evaluations.
        - Updated eval_population array with new evaluation scores.
        - offsprings_idx (list): Indices of offspring individuals added to the population.
        """
        parents = [population[idx] for idx in parents_idx]
        
        num_features = len(x)
        num_individuals = len(parents)
        
        for i in range(0, num_individuals, 2):
            if np.random.rand() <= pc and i + 1 < num_individuals:
                parent1 = parents[i].copy()
                parent2 = parents[i + 1].copy()
                
                crossover_points = np.random.choice(range(num_features), size=2, replace=False)
                
                for point in crossover_points:
                    if self.columns[point] not in self.constraints:
                        value_p1 = parent1[point]
                        value_p2 = parent2[point]

                        parent1[point] = generate_random_value(value_p2, x[point], value_p1)
                        parent2[point] = generate_random_value(value_p1, x[point], value_p2)

                        parent1[point + num_features], parent2[point + num_features] = parent2[point + num_features], parent1[point + num_features]

                population, eval_population, offsprings_idx = self.elitist_check_eval_offspring(
                    [parent1, parent2], [i, i + 1], population, x, eval_population
                )
                    
        return population, eval_population, offsprings_idx

    def mutate(self, population, x, parents_idx, eval_population, pm, elitist=False):
        """
        Apply mutation to selected offspring individuals, adjusting feature values and weights with a specified mutation probability.
        The function optionally replaces parents with mutated offspring based on fitness evaluations and applies elitist selection
        if specified.

        Parameters:
        - population (list): The current population of individuals, where each individual is a list of feature values.
        - x (list): The original instance against which counterfactuals are generated.
        - parents_idx (list): Indices of parent individuals selected for potential mutation and replacement.
        - eval_population (np.ndarray): Evaluation scores for the current population.
        - pm (float): Mutation probability, dictating the likelihood of mutating an offspring.
        - elitist (bool): If True, elitist selection is applied to retain the fittest individuals, even if offspring are mutated.

        Returns:
        - tuple: 
        - Updated population with mutated offspring replacing parents if they perform better.
        - Updated eval_population array with new evaluation scores.
        """
        num_offsprings = len(parents_idx)
        num_features = len(x)
        num_individuals = len(population)
        weight = 1 / (2 * num_features)

        offsprings = [population[idx] for idx in parents_idx]

        for i in range(num_offsprings):
            if np.random.rand() <= pm:
                mutation_points = np.random.choice(range(num_features), size=2, replace=False)

                for point in mutation_points:
                    # If no constraints on feature, mutate by averaging feature values from two random individuals
                    if self.dataset.columns[point] not in self.constraints:
                        if isinstance(offsprings[i][point], (np.integer, np.floating)):
                            idx1, idx2 = np.random.choice(range(num_individuals), size=2, replace=False)
                            offsprings[i][point] = np.mean([population[idx1][point], population[idx2][point]])
                        else:
                            # For categorical data, randomly select feature value from another individual
                            random_idx = np.random.randint(num_individuals)
                            offsprings[i][point] = population[random_idx][point]
                    
                    # Enforce 'up'/'down' constraints as applicable
                    if self.dataset.columns[point] in self.partial_constraints.keys():
                        constraint = self.partial_constraints[self.dataset.columns[point]]
                        original_value = population[parents_idx[0]][point]

                        if constraint == 'up' and offsprings[i][point] < original_value:
                            offsprings[i][point] = original_value
                        elif constraint == 'down' and offsprings[i][point] > original_value:
                            offsprings[i][point] = original_value

            if np.random.rand() <= pm:
                weight_points = np.random.choice(range(num_features), size=2, replace=False)
                change_amount = np.random.uniform(0, weight)
                offsprings[i][weight_points[0] + num_features] += change_amount
                offsprings[i][weight_points[1] + num_features] -= change_amount

        if elitist:
            population, eval_population, _ = self.elitist_check_eval_offspring(offsprings, parents_idx, population, x, eval_population)
        else:
            population, eval_population = self.check_eval_offspring(offsprings, parents_idx, population, x, eval_population)

        return population, eval_population

    def filter_population(self, mutated_population, x, eval_population):
        """
        Filters out individuals from the mutated population that match the target class of the input instance 'x'.
        Individuals whose predicted class does not match 'x' are retained in the population.

        :param mutated_population: Population of mutated individuals with features and weights.
        :param x: The original instance with the target class for comparison.
        Returns:
        - tuple (np.ndarray, np.ndarray): 
        - A NumPy array of the filtered population, retaining only individuals with differing predictions from the original target.
        - A NumPy array of updated evaluation scores after removing individuals with unchanged predictions.
        """
        filtered_population = []
        
        results = inference(self.model, torch.tensor(mutated_population[:, :len(x) - 1], dtype=torch.float32))

        idx_to_remove = []

        for idx, result in enumerate(results):
            if result != x['target']:
                filtered_population.append(mutated_population[idx])
            else:
                idx_to_remove.append(idx)

        eval_population = np.delete(eval_population, idx_to_remove)

        return np.array(filtered_population), np.array(eval_population)
    
    def filter_encoded_population(self, mutated_population, x, eval_population):
        """
        Filter the mutated population to retain only individuals that produce a different prediction from the original instance.

        This method encodes the categorical features of the mutated population, applies a pre-trained model to make predictions, 
        and filters out individuals that maintain the original class label. It ensures that only individuals whose predictions 
        differ from the original target are retained, along with updating the evaluation scores accordingly.

        Parameters:
        - mutated_population (np.ndarray): The mutated population as a NumPy array, with each row representing an individual.
        - x (dict): A dictionary containing the original instance's features and 'target' class label.
        - eval_population (np.ndarray): An array of evaluation scores corresponding to the individuals in the mutated population.

        Returns:
        - tuple (np.ndarray, np.ndarray): 
        - A NumPy array of the filtered population, retaining only individuals with differing predictions from the original target.
        - A NumPy array of updated evaluation scores after removing individuals with unchanged predictions.
        """
        filtered_population = []
        
        df = pd.DataFrame(data=[np.zeros(len(self.encoded_columns))], columns=self.encoded_columns)
        
        encoded_population = pd.DataFrame(data=mutated_population[:, :len(self.columns)], columns=self.columns)
        
        encoded_population = pd.get_dummies(encoded_population).astype(float)
        
        df = pd.concat([df] * len(encoded_population), ignore_index=True)
        
        df.update(encoded_population)
        
        data = df.values
        
        results = inference(self.model, torch.tensor(data, dtype=torch.float32))
        
        idx_to_remove = []

        for idx, result in enumerate(results):
            if result != x['target']:
                filtered_population.append(mutated_population[idx])
            else:
                idx_to_remove.append(idx)
        
        eval_population = np.delete(eval_population, idx_to_remove)

        return np.array(filtered_population), np.array(eval_population)

    def __call__(self):
        """
        Executes the AFWGE algorithm to generate counterfactual explanations for each instance in the dataset.
        The method evolves a population of individuals through several generations using genetic operations 
        (selection, crossover, mutation) to find counterfactuals that are close to the original instance but 
        belong to a different target class.

        :return: A list of tuples, where each tuple contains the original instance and its corresponding counterfactual.
        """
        counterfactuals = []

        for idx in range(self.dataset.shape[0]):
            x = self.dataset.iloc[idx].values[:-1]

            population = []
            eval_population = []

            terminationCounter = 0
            best_fitness = float('inf')

            pc = self.pc
            pm = self.pm

            elitist = False

            for i in range(self.generations):
                population, eval_population = self.generate_population(self.k - len(population), idx, population, eval_population)

                parents_idx = self.get_parents_idx(eval_population)
                
                if int(0.6 * self.generations) < i:
                    elitist = True
                    pc += pc * 0.1 
                    pm -= pm * 0.1

                if int(0.7 * self.generations) < i:
                    population, eval_population, offsprings_idx = self.interpolates_crossover(population, x, parents_idx, eval_population, pc)
                else:
                    population, eval_population, offsprings_idx = self.crossover(population, x, parents_idx, eval_population, pc, elitist)
                
                population, eval_population = self.mutate(population, x, offsprings_idx, eval_population, pm, elitist)
                
                if self.encoded:
                    population, eval_population = self.filter_encoded_population(population, self.dataset.iloc[idx], eval_population)
                else:
                    population, eval_population = self.filter_population(population, self.dataset.iloc[idx], eval_population)

                idx_sorted = sorted(range(len(eval_population)), key=lambda i: eval_population[i])

                if len(idx_sorted) > 0 and eval_population[idx_sorted[0]] < best_fitness:
                    best_fitness = eval_population[idx_sorted[0]]
                    terminationCounter = 0
                else:
                    terminationCounter += 1

                if terminationCounter >= 5:
                    break

                if not elitist and len(population) == self.k and i != self.generations - 1:
                    j = random.randint(-self.k // 2, self.k - 2)
                    population = population[idx_sorted[:j]]
                    eval_population = eval_population[idx_sorted[:j]]

            length = int(self.q * len(idx_sorted))
            for i in range(length):
                counterfactuals.append((x, population[idx_sorted[i]]))

        return counterfactuals
