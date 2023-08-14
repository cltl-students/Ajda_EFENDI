import pandas as pd

# Load the dataset
df = pd.read_csv('training_dataset_use')

# Group the documents by level and concatenate the descriptions
grouped_df = df.groupby('eqf_level_id')['description'].apply('\n'.join).reset_index()

# Iterate over the groups and save each level's document to a separate file
for _, row in grouped_df.iterrows():
    level_id = row['eqf_level_id']
    description = row['description']

    # Save the level document to a separate file
    filename = f'{level_id}'
    with open(filename, 'w') as file:
        file.write(description)

    print(f'Saved level {level_id} document to {filename}')
    #import pandas as pd

# Create an empty DataFrame
merged_df = pd.DataFrame(columns=['eqf_level_id', 'description'])

# Iterate over each document
for doc_num in range(1, 9):
    # Read the document content
    with open(f'{doc_num}', 'r') as file:
        content = file.read()

    # Create a DataFrame for the current document
    doc_df = pd.DataFrame({'eqf_level_id': [doc_num], 'description': [content]})

    # Append the document DataFrame to the merged DataFrame
    merged_df = merged_df.append(doc_df, ignore_index=True)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_training.txt', index=False)

# SECOND TRAINING FOR SECOND DEV DATA
import pandas as pd

# Load the dataset
df = pd.read_csv('sec_training_dataset_use')

# Group the documents by level and concatenate the descriptions
grouped_df = df.groupby('eqf_level_id')['description'].apply('\n'.join).reset_index()

# Iterate over the groups and save each level's document to a separate file
for _, row in grouped_df.iterrows():
    level_id = row['eqf_level_id']
    description = row['description']

    # Save the level document to a separate file
    filename = f'{level_id}'
    with open(filename, 'w') as file:
        file.write(description)

    print(f'Saved level {level_id} document to {filename}')
    #import pandas as pd

# Create an empty DataFrame
merged_df = pd.DataFrame(columns=['eqf_level_id', 'description'])

# Iterate over each document
for doc_num in range(1, 9):
    # Read the document content
    with open(f'{doc_num}', 'r') as file:
        content = file.read()

    # Create a DataFrame for the current document
    doc_df = pd.DataFrame({'eqf_level_id': [doc_num], 'description': [content]})

    # Append the document DataFrame to the merged DataFrame
    merged_df = merged_df.append(doc_df, ignore_index=True)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('sec_merged_training.txt', index=False)

