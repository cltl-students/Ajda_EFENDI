# this dscipt is to align the datasets by column. Some datasets had more columns than necessary and
# in this code they are removed and renamed
import pandas as pd
# read the dataset
pathswedish='/Users/ajdaefendi/Downloads/edulevel_classifier-main/data/dataset_myh.csv'
dfswedish=pd.read_csv(pathswedish)
# combine the descriptive columuns
headers=dfswedish['knowledge']+dfswedish['skills']+dfswedish['competence']
# add the combined columns into a new column
dfswedish['description']=headers
# drop the unnecessary columns
dfswedish=dfswedish.drop(['Unnamed: 0','file','uri','knowledge','skills','competence'], axis=1)
# write the data into a new file
dfswedish.to_csv('SWEDISH1')
# #
# #
pathswedish='/Users/ajdaefendi/Downloads/edulevel_classifier-main/data/dataset_myh2.csv'
dfswedish=pd.read_csv(pathswedish)
# combine the descriptive columuns
headers=dfswedish['knowledge']+dfswedish['skills']+dfswedish['competence']
# add the combined columns into a new column
dfswedish['description']=headers
# drop the unnecessary columns
dfswedish=dfswedish.drop(['file','uri','knowledge','skills','competence'], axis=1)
# write the data into a new file
dfswedish.to_csv('SWEDISH2')

# #
pathswedish='/Users/ajdaefendi/Downloads/edulevel_classifier-main/data/dataset_myh3.csv'
dfswedish=pd.read_csv(pathswedish)
# combine the descriptive columuns
headers=dfswedish['knowledge']+dfswedish['skills']+dfswedish['competence']
# add the combined columns into a new column
dfswedish['description']=headers
# drop the unnecessary columns
dfswedish=dfswedish.drop(['file','uri','knowledge','skills','competence'], axis=1)
# write the data into a new file
dfswedish.to_csv('SWEDISH3')

# #
# #
# read the dataset
pathdutch='/Users/ajdaefendi/Downloads/edulevel_classifier-main/data/dataset_nlqf.csv'
dfdutch=pd.read_csv(pathdutch)

# drop the unnecessary columns
dfdutch=dfdutch.drop(['file','uri'], axis=1)

# create a new file for the dataset
dfdutch.to_csv('NEW_DUTCH')

swedish2=s2='/Users/ajdaefendi/Downloads/edulevel_classifier-main/data/dataset_myh2.csv'
dfs2=pd.read_csv(swedish2)
headers=dfs2['knowledge']+dfs2['skills']+dfs2['competence']
dfs2['description']=headers
dfs2=dfs2.drop(['file','uri','knowledge','skills','competence'], axis=1)
dfs2.to_csv('NEWSWEDISH2')
pathdutch='/Users/ajdaefendi/Downloads/edulevel_classifier-main/NEW_DUTCH'
dfdutch=pd.read_csv(pathdutch)
print(len(dfdutch))
print('DUTCHdes',len(dfdutch['description']),'DUTCHLEVEL',len(dfdutch['eqf_level_id']))
swedish2=s2='/Users/ajdaefendi/Downloads/edulevel_classifier-main/NEWSWEDISH'
dfs2=pd.read_csv(swedish2)
print('SWEDISHdesc', len(dfs2['description']),'SWLEVEL',len(dfs2['eqf_level_id']))
# print(dfs2['eqf_level_id'][:200])


# Check the new datasets
dataducth='/Users/ajdaefendi/Downloads/edulevel_classifier-main/NEW_DUTCH'
dataswedish='/Users/ajdaefendi/Downloads/edulevel_classifier-main/NEWSWEDISH'
dfd=pd.read_csv(dataducth, sep=',')
dfs=pd.read_csv(dataswedish, sep=',')
#
# check the length of the new datasets
print(len(dfd))
print(len(dfs))

# add all same levels together in one column
for i in dfs['eqf_level_id']:
    print(i)
print(set(dfs['eqf_level_id']))
print(set(dfd['eqf_level_id']))
for i in dfs['eqf_level_id']:
    if i == 3:
        dfs['level3']=dfs['description']
    elif i == 5:
        dfs['level5']=dfs['description']
    elif i== 4:
        dfs['level4']=dfs['description']
    else:
        dfs['level6']=dfs['description']
# print(dfs['level6'])
# create a new file as level-separated data
dfs.to_csv(('level_sep'))
# check the new data
sep='/Users/ajdaefendi/Downloads/edulevel_classifier-main/level_sep'
dfl=pd.read_csv(sep, sep=',')
print(dfl['level5'])
