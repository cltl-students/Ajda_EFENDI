# README
It is important to note that the code is developed and the errors are fixed by using both "Stackoverflow" and "ChatGPT". The fixed errors are manually check to provide further reliance.


Classification Code and Dataset Generation

This repository contains code for various classification experiments using different datasets and systems. The code files are organized by number for a systematic approach to dataset generation and classification.

Dataset Generation

0__align_datasets.py - Aligns datasets per column
1__latvia_preprocess.py - Webscrapes the latvian dataset and saves it to a file
2__combine_training_datasets.py - Merges the separate datasets depending on the purpose of training. It creates non-dutch training set (named as "training_dataset.txt") and all-datasets combined (named as "eval_test") datasets.
3__3__development_data.py - it creates 2 development datasets per experiment mentioned in the report. The non-dutch development dataset is extracted from "training_dataset.txt" (as it has no dutch instances) and it is called "development_dataset.txt"  and the all-datasets-combined development dataset is extracted from the "eval_test" dataset as it is the compiled form of all datasets used in this research. 
4__create_trainingset_all_datasets_combined.py - Splits the unified dataset into training and testing sets for the all-datasets-combined system.
5__merge_dataset.py - it creates the final versions of the datasets. That is, it merges the columns of the training systems and creates one compiled document per class. Then, it saves them to separate files: for the non-dutch training system it creates "merged_training.txt" and for the all datasets combined system it creates "sec_merged_training.txt" file which are compiled per class and the development datasets are extracted.

Classification Experiments

dutch_as_eval_testing_TF-IDF - Evaluates the non-Dutch training system on the Dutch dataset.
evaluation_set_wdutch - Performs classification experiments using the all-datasets-combined system.
Before running evaluation_set_wdutch, ensure that 4_create_trainingset_all_datasets_combined has been executed to prepare the necessary dataset.
logreg - evaluates the Machine Learning approach in both (non-dutch and all-datasets-combined experiments).
It should be highlighted that for desired experiments correct datasets should be inputted to these systems.

The datasets to input
training:
  -merged_dataset.txt - includes Swedish, Latvian and Maltese datasets (Non-Dutch training dataset) 
  - wdutch_train.txt - includes Swedish, Latvian, Maltese and Dutch datasets (All-datasets-combined dataset)
testing;
  -NEW_DUTCH - only dutch dataset (testing performance of dutch data on other EQF descriptions)
  -wdutch_test.txt - includes Swedish, Latvian, Maltese and Dutch datasets (Investigating whether incorporating instances from all datasets in both the training and test data leads to improved performance compared to the non-Dutch training system.)

How to Use

Run the dataset generation scripts in the order of the provided numbers.
Once the datasets are prepared, you can run the classification scripts:
Execute dutch_as_eval_testing_TF-IDF for evaluating non-Dutch training on the Dutch dataset.
Execute evaluation_set_wdutch to perform experiments with the all-datasets-combined system.
Execute logreg system.
Make sure to adjust any file paths or settings as needed for your specific environment.

Requirements

Python 3.x
Required Python libraries (e.g., pandas, scikit-learn)
altgraph @ file:///System/Volumes/Data/SWE/Apps/DT/BuildRoots/BuildRoot7/ActiveBuildRoot/Library/Caches/com.apple.xbs/Sources/python3/python3-133.100.1.1/altgraph-0.17.2-py2.py3-none-any.whl
async-generator==1.10
attrs==22.2.0
BeautifulRequests==0.2
beautifulsoup4==4.12.0
blis==0.7.9
bs4==0.0.1
catalogue==2.0.8
certifi==2022.12.7
chardet==3.0.4
charset-normalizer==3.1.0
click==8.1.3
confection==0.0.4
contourpy==1.1.0
cycler==0.11.0
cymem==2.0.7
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl#sha256=0964370218b7e1672a30ac50d72cdc6b16f7c867496f1d60925691188f4d2510
exceptiongroup==1.1.1
filelock==3.10.7
fonttools==4.40.0
future @ file:///System/Volumes/Data/SWE/Apps/DT/BuildRoots/BuildRoot7/ActiveBuildRoot/Library/Caches/com.apple.xbs/Sources/python3/python3-133.100.1.1/future-0.18.2-py3-none-any.whl
googletrans==4.0.0rc1
h11==0.9.0
h2==3.2.0
hpack==3.0.0
hstspreload==2023.1.1
httpcore==0.9.1
httpx==0.13.3
huggingface-hub==0.13.3
hyperframe==5.2.0
idna==2.10
importlib-resources==5.12.0
iniconfig==2.0.0
Jinja2==3.1.2
joblib==1.2.0
kiwisolver==1.4.4
langcodes==3.3.0
langid==1.1.6
lxml==4.9.2
macholib @ file:///System/Volumes/Data/SWE/Apps/DT/BuildRoots/BuildRoot7/ActiveBuildRoot/Library/Caches/com.apple.xbs/Sources/python3/python3-133.100.1.1/macholib-1.15.2-py2.py3-none-any.whl
MarkupSafe==2.1.2
matplotlib==3.7.1
mpmath==1.3.0
multidict==6.0.4
murmurhash==1.0.9
networkx==3.0
nltk==3.8.1
numpy==1.24.2
outcome==1.2.0
packaging==23.0
pandas==1.5.3
pathy==0.10.1
Pillow==9.5.0
pluggy==1.0.0
preshed==3.0.8
psutil==5.9.5
pydantic==1.10.7
pyparsing==3.1.0
PySocks==1.7.1
pytest==7.2.2
python-dateutil==2.8.2
pytz==2023.3
PyYAML==6.0
regex==2023.3.23
requests==2.28.2
rfc3986==1.5.0
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
selenium==4.9.1
six @ file:///System/Volumes/Data/SWE/Apps/DT/BuildRoots/BuildRoot7/ActiveBuildRoot/Library/Caches/com.apple.xbs/Sources/python3/python3-133.100.1.1/six-1.15.0-py2.py3-none-any.whl
sklearn==0.0.post1
smart-open==6.3.0
sniffio==1.3.0
sortedcontainers==2.4.0
soupsieve==2.4
spacy==3.5.2
spacy-legacy==3.0.12
spacy-loggers==1.0.4
srsly==2.4.6
sympy==1.11.1
thinc==8.1.9
threadpoolctl==3.1.0
tokenizers==0.13.2
tomli==2.0.1
torch==2.0.0
tqdm==4.65.0
transformers==4.27.4
trio==0.22.0
trio-websocket==0.10.2
typer==0.7.0
typing_extensions==4.5.0
urllib3==1.26.15
vcrpy==4.2.1
wasabi==1.1.1
wrapt==1.15.0
wsproto==1.2.0
yarl==1.8.2
zipp==3.15.0
