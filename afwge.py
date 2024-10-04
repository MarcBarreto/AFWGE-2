import torch
import random
import numpy as np
import pandas as pd
from mlp import inference

class AFWGE:
    def __init__(self, model, scaler, dataset, constraints=[None], select=2, k=100, generations=10, q=0.7, pc=0.7, pm=0.2):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.constraints = constraints
        self.select = select
        self.k = k
        self.generations = generations
        self.q = q
        self.pc = pc
        self.pm = pm

    def generate_population(self, length, individual_idx, population=[]):
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

        for idx, column in enumerate(columns):
            if pd.api.types.is_numeric_dtype(self.dataset[column]):
                if column not in self.constraints:
                    if min_features.iloc[idx] == int(min_features.iloc[idx]) and max_features.iloc[idx] == int(max_features.iloc[idx]):
                        new_population.append(np.random.randint(int(min_features.iloc[idx]), int(max_features.iloc[idx]) + 1, length))
                    else:
                        new_population.append(np.random.uniform(min_features.iloc[idx], max_features.iloc[idx], length))
                else:
                    new_population.append([self.dataset.iloc[individual_idx].iloc[idx]] * length)
            else:
                if column not in self.constraints:
                    categories = self.dataset[column].unique()
                    new_population.append(np.random.choice(categories, length))
                else:
                    category = self.dataset.iloc[individual_idx].iloc[idx]
                    aux = []
                    for i in range(length):
                        aux.append(category)
                    new_population.append(aux)

        for column in columns:
            new_population.append([weight] * length)

        new_population = np.stack(new_population, axis=-1)

        if len(population) > 0:
            return np.concatenate((population, new_population), axis=0)

        return new_population

    def matching_distance(self, x, c):
        """
        Calculates the weighted distance between an original instance 'x' and a counterfactual instance 'c'.
        Numerical and categorical features are treated differently, using feature-specific weights.

        :param x: The original instance.
        :param c: The counterfactual instance, including feature weights.
        :return: The weighted distance between 'x' and 'c'.
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

    def get_parents_idx(self, scores):
        """
        Selects the best individuals from the population based on their evaluation scores.
        The number of selected parents is defined by the parameter 'select'.

        :param scores: Array of evaluation scores for the current population.
        :return: Indices of the top-scoring individuals to be selected as parents.
        """
        sorted_indices = np.argsort(scores)
        return sorted_indices[:self.select]

    def check_eval_offspring(self, offspring, parents_idx, population, x, eval_population):
        """
        Evaluates the fitness of an offspring by calculating its distance to the original instance 'x'. 
        If the offspring has better fitness (lower distance) than a parent, the parent is replaced by the offspring.

        :param offspring: The new generated offspring.
        :param parents_idx: Indices of the parent individuals in the population.
        :param population: Current population of individuals.
        :param x: The original instance.
        :param eval_population: Evaluation scores of the current population.
        :return: Updated population with potentially replaced individuals.
        """
        for idx in parents_idx:
            offspring_distance = self.matching_distance(x, offspring)
            if offspring_distance < eval_population[idx]:
                population[idx] = offspring
                eval_population[idx] = offspring_distance
                break
        return population

    def crossover(self, population, x, parents_idx, eval_population):
        """
        Performs crossover between selected parent individuals to generate offspring.
        Features are swapped between two parents based on randomly selected crossover points.
        
        :param population: Current population of individuals.
        :param x: The original instance.
        :param parents_idx: Indices of selected parent individuals for crossover.
        :param eval_population: Evaluation scores of the current population.
        :return: Updated population and the list of generated offspring.
        """
        offsprings = []
        parents = [population[idx] for idx in parents_idx]
        num_features = len(x)
        num_individuals = len(parents)

        for i in range(0, num_individuals, 2):
            if np.random.rand() <= self.pc and i + 1 < num_individuals:
                parent1 = parents[i].copy()
                parent2 = parents[i+1].copy()
                crossover_points = np.random.choice(range(num_features), size=2, replace=False)

                for point in crossover_points:
                    if self.dataset.columns[point] not in self.constraints:
                        parent1[point], parent2[point] = parent2[point], parent1[point]
                        parent1[point + num_features], parent2[point + num_features] = parent2[point + num_features], parent1[point + num_features]

                population = self.check_eval_offspring(parent1, [i, i + 1], population, x, eval_population)
                population = self.check_eval_offspring(parent2, [i, i + 1], population, x, eval_population)
                offsprings.append(parent1)
                offsprings.append(parent2)
            else:
                offsprings.append(population[i])
                if i + 1 < num_individuals:
                    offsprings.append(population[i+1])

        return population, offsprings

    def mutate(self, population, x, parents_idx, offsprings, eval_population):
        """
        Applies mutation to offspring individuals, modifying feature values and corresponding weights with a given mutation probability.
        If a mutated offspring is better than a parent, it replaces the parent in the population.

        :param population: Current population of individuals.
        :param x: The original instance.
        :param parents_idx: Indices of selected parent individuals that might be replaced.
        :param offsprings: List of offspring generated by crossover.
        :param eval_population: Evaluation scores of the current population.
        :return: Updated population after mutation.
        """
        num_offsprings = len(offsprings)
        num_features = len(x)
        num_individuals = len(population)
        weight = 1 / (2 * num_features)

        for i in range(num_offsprings):
            if np.random.rand() <= self.pm:
                mutation_points = np.random.choice(range(num_features), size=2, replace=False)

                for point in mutation_points:
                    if self.dataset.columns[point] not in self.constraints:
                        if isinstance(offsprings[i][point], (np.integer, np.floating)):
                            idx1, idx2 = np.random.choice(range(num_individuals), size=2, replace=False)
                            offsprings[i][point] = np.mean([population[idx1][point], population[idx2][point]])
                        else:
                            random_idx = np.random.randint(num_individuals)
                            offsprings[i][point] = population[random_idx][point]

                population = self.check_eval_offspring(offsprings[i], parents_idx, population, x, eval_population)

            if np.random.rand() <= self.pm:
                weight_points = np.random.choice(range(num_features), size=2, replace=False)
                change_amount = np.random.uniform(0, weight)
                offsprings[i][weight_points[0] + num_features] += change_amount
                offsprings[i][weight_points[1] + num_features] -= change_amount

                population = self.check_eval_offspring(offsprings[i], parents_idx, population, x, eval_population)

        return population

    def filter_population(self, mutated_population, x):
        """
        Filters out individuals from the mutated population that match the target class of the input instance 'x'.
        Individuals whose predicted class does not match 'x' are retained in the population.

        :param mutated_population: Population of mutated individuals with features and weights.
        :param x: The original instance with the target class for comparison.
        :return: Filtered population of individuals that do not match the target class of 'x'.
        """
        filtered_population = []
        results = inference(self.model, torch.tensor(self.scaler.transform(mutated_population[:, :len(x) - 1]), dtype=torch.float32))

        for idx, result in enumerate(results):
            if result != x['target']:
                filtered_population.append(mutated_population[idx])

        return np.array(filtered_population)

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
            terminationCounter = 0
            best_fitness = float('inf')

            for i in range(self.generations):
                population = self.generate_population(self.k - len(population), idx, population)
                eval_population = np.array([self.matching_distance(x, individual) for individual in population])

                parents_idx = self.get_parents_idx(eval_population)
                population, offsprings = self.crossover(population, x, parents_idx, eval_population)
                population = self.mutate(population, x, parents_idx, offsprings, eval_population)
                population = self.filter_population(population, self.dataset.iloc[idx])

                eval_population = np.array([self.matching_distance(x, individual) for individual in population])
                idx_sorted = sorted(range(len(eval_population)), key=lambda i: eval_population[i])

                if eval_population[idx_sorted[0]] < best_fitness:
                    best_fitness = eval_population[idx_sorted[0]]
                    terminationCounter = 0
                else:
                    terminationCounter += 1

                if terminationCounter >= 5:
                    break

                if len(population) == self.k and i != self.generations - 1:
                    j = random.randint(-self.k // 2, self.k - 2)
                    population = population[idx_sorted[:j]]
                    eval_population = eval_population[idx_sorted[:j]]

            length = int(self.q * len(idx_sorted))
            for i in range(length):
                counterfactuals.append((x, population[idx_sorted[i]]))

        return counterfactuals
