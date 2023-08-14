
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data into a pandas DataFrame (replace this with your data loading code)
data = pd.read_csv('merged_training.txt')
data2=pd.read_csv('sec_merged_training.txt')

# Define the features (X) and the target (y) columns
X = data.drop(columns=['descriptions'])
y = data['descriptions']


# Split the data into training and testing sets (70% for training, 30% for testing)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save the training and testing sets to separate CSV files
train_data.to_csv('wdutch_train.txt', index=False)
test_data.to_csv('wdutch_test.txt', index=False)

# all-datasets-combined system merge columns
# Load the dataset
df = pd.read_csv('wdutch_train.txt')

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
merged_df.to_csv('wdutch_train.txt', index=False)
