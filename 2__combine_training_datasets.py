# This code is to compile/merge all datasets for the training. Before, the datasets were separated from each other.
import pandas as pd
df1=pd.read_csv('swedish_dataset')
df1=df1.drop(columns=['Unnamed: 0.1','Unnamed: 0'], axis=1)
df2=pd.read_csv('latvian_dataset')

# https://towardsdatascience.com/python-pandas-tricks-3-best-methods-4a909843f5bc#1d6f
new=pd.merge(df1, df2, how='outer')
new.to_csv('nondutch_training_dataset')


two=pd.read_csv('nondutch_training_dataset')
swla=two.drop(columns=['Unnamed: 0'], axis=1)
# https://stackoverflow.com/questions/41719259/how-to-remove-numbers-from-string-terms-in-a-pandas-dataframe
swla['description'] = swla['description'].str.replace('\d+', '')


df3=pd.read_csv('dataset_malta.csv')
df3=df3.drop(columns=['file','uri','isced_code'])
df3['description'] = df3['description'].str.replace('\d+', '')

print(df3)

combined=pd.merge(swla, df3, how='outer')
combined.to_csv(('training_dataset.txt'))

test_data = pd.read_csv('NEW_DUTCH')
train_data=pd.read_csv('training_dataset.txt')
# Concatenate the training and test datasets
merged_data = pd.concat([train_data, test_data], ignore_index=True)

# Write the merged dataset to a new file
merged_data.to_csv('eval_test', index=False)
