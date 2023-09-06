# GA_dimensionality_reduction
genetic algorithm for dimensionality reduction

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate artificial data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the evaluation function (using KNN as a classifier)
def evaluate(individual):
    # Get indices of selected features
    indices = [i for i, feature in enumerate(individual) if feature]
    # Extract selected features
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]
    # Train a classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_selected, y_train)
    # Predict on the test set
    y_pred = clf.predict(X_test_selected)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Return the fitness value (accuracy)
    return accuracy

# Genetic Algorithm parameters
population_size = 100
chromosome_length = X.shape[1]
mutation_rate = 0.1
generations = 50

# Create the initial population
population = np.random.randint(0, 2, size=(population_size, chromosome_length)).astype(bool)

# Run the Genetic Algorithm
for generation in range(generations):
    # Evaluate the fitness of individuals
    fitness_values = np.array([evaluate(individual) for individual in population])
    
    # Select parents for reproduction (roulette wheel selection)
    fitness_sum = np.sum(fitness_values)
    probabilities = fitness_values / fitness_sum
    cumulative_probabilities = np.cumsum(probabilities)
    
    offspring = []
    for _ in range(population_size):
        # Select two parents
        parent1_idx = np.searchsorted(cumulative_probabilities, np.random.random())
        parent2_idx = np.searchsorted(cumulative_probabilities, np.random.random())
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        
        # Perform crossover (single-point crossover)
        crossover_point = np.random.randint(1, chromosome_length)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        
        # Perform mutation (bit-flip mutation)
        for i in range(chromosome_length):
            if np.random.random() < mutation_rate:
                child[i] = not child[i]
        
        offspring.append(child)
    
    # Replace the old population with the offspring
    population = np.array(offspring)

# Select the best individual from the final population
best_individual_idx = np.argmax([evaluate(individual) for individual in population])
best_individual = population[best_individual_idx]
selected_features = [i for i, feature in enumerate(best_individual) if feature]

# Print the selected features
print("Selected Features:", selected_features)
```
