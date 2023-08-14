# This code creates the development datasets
import pandas as pd
# Load the training dataset
dataset_A = pd.read_csv('training_dataset.txt')

# Initialize an empty DataFrame for the development dataset
development_dataset = pd.DataFrame()

# Iterate over levels 1 to 7
for level in range(1, 9):
    # Filter the training dataset for the current level
    level_instances = dataset_A[dataset_A['eqf_level_id'] == level].sample(n=50, random_state=42)

    # Add the selected instances to the development dataset
    development_dataset = development_dataset.append(level_instances)

    # Remove the selected instances from the training dataset
    dataset_A = dataset_A.drop(level_instances.index)

# Save the development dataset
development_dataset.to_csv('development_dataset.txt', index=False)

# Save the updated training dataset without the instances in the development dataset
dataset_A.to_csv('training_dataset_use', index=False)

print(development_dataset['eqf_level_id'].value_counts())


# SECOND DEV
import pandas as pd

# Load the training dataset
dataset_A = pd.read_csv('eval_test')

# Initialize an empty DataFrame for the development dataset
development_dataset = pd.DataFrame()

# Iterate over levels 1 to 7
for level in range(1, 9):
    # Filter the training dataset for the current level
    level_instances = dataset_A[dataset_A['eqf_level_id'] == level].sample(n=50, random_state=42)

    # Add the selected instances to the development dataset
    development_dataset = development_dataset.append(level_instances)

    # Remove the selected instances from the training dataset
    dataset_A = dataset_A.drop(level_instances.index)

# Save the development dataset
development_dataset.to_csv('sec_development_dataset.txt', index=False)

# Save the updated training dataset without the instances in the development dataset
dataset_A.to_csv('sec_training_dataset_use', index=False)

print(development_dataset['eqf_level_id'].value_counts())