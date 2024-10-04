# AFWGE-2: Adaptive Genetic Algorithm with Embedded Feature Weights for Counterfactual Explanations

This repository contains the implementation of the an adaptative counterfactual explanation algorithm described in the paper [Counterfactual Explanation of AI Models Using an Adaptive Genetic Algorithm With Embedded Feature Weights](https://ieeexplore.ieee.org/document/10536083), written by Ebtisam AlJalaud and Manar Hosny.

## Table of Contents
- [Introduction](#introduction)
- [What is XAI?](#what-is-xai)
- [What are Counterfactual Explanations?](#what-are-counterfactual-explanations)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Others](#others)
## Introduction

This project provides an implementation of a counterfactual generation algorithm that explains AI model decisions by finding alternative inputs that could have led to different outcomes. The algorithm is built on an adaptive genetic algorithm with feature weights that adjusts to the importance of features during the optimization process. The code is written in Python and uses PyTorch for the machine learning components.

## What is XAI?

**Explainable Artificial Intelligence (XAI)** refers to methods and techniques that make the decisions of AI systems understandable to humans. The goal of XAI is to increase transparency, trust, and fairness in AI models, particularly in critical applications such as healthcare, finance, and autonomous systems. By making AI decisions more interpretable, XAI helps address the "black-box" problem inherent in many complex models like neural networks and ensemble methods.

## What are Counterfactual Explanations?

**Counterfactual explanations** describe how a model's output can be changed by altering the input. For instance, a counterfactual might answer the question: "What would need to change in this input for the model to predict a different class?" These explanations are valuable because they provide actionable insights for users who want to understand how a model's decisions are influenced by its inputs.

In this project, the counterfactuals are generated using an adaptive genetic algorithm, which optimizes the search for alternative inputs while accounting for the feature importance embedded in the dataset.

## Setup

To run this project, follow the steps below:

1. **Create a Conda Environment**:
   ```bash
   conda create --name afwge-env python=3.9
   conda activate afwge-env
   ```
2. **Install Dependencies: Install the required libraries using the `requirements.txt` file**:
   ```bash
   pip install -r requirements.txt

## Usage

To run the project and generate counterfactuals for the Iris dataset, execute the following command:
  ```bash
  python3 main.py
  ```
This will train a neural network on the Iris dataset and use the Adaptive Genetic Algorithm (AFWGE) to generate counterfactual explanations. The generated counterfactuals will be saved to a file called `counterfactuals.csv` in the project directory.

## Project Structure
- `main.py`: The main script that runs the model training and counterfactual generation process.
- `afwge.py`: Contains the implementation of the adaptive genetic algorithm for counterfactual generation.
- `mlp.py`: Defines the Multi-Layer Perceptron (MLP) model used to classify the Iris dataset.
- `dataset.py`: Handles the preprocessing of the Iris dataset and converts it into PyTorch tensors.
- `utils.py`: Contains utility functions, such as exporting counterfactuals to a CSV file.

## Others
For more details, read the full [article](https://ieeexplore.ieee.org/document/10536083).
