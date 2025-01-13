# ASSIGNMENT

## 1. Make sure that you have python 3.11 (I used 3.11.8) installed on your computer. If not, it is going to be a problem.
## 2. Make sure that you have the following packages installed:
- absl-py==2.1.0
- astunparse==1.6.3
- certifi==2024.12.14
- charset-normalizer==3.4.1
- click==8.1.8
- colorama==0.4.6
- contourpy==1.3.1
- cycler==0.12.1
- flatbuffers==24.12.23
- fonttools==4.55.3
- gast==0.6.0
- google-pasta==0.2.0
- grpcio==1.69.0
- h5py==3.12.1
- idna==3.10
- joblib==1.4.2
- keras==3.8.0
- kiwisolver==1.4.8
- libclang==18.1.1
- Markdown==3.7
- markdown-it-py==3.0.0
- MarkupSafe==3.0.2
- matplotlib==3.10.0
- mdurl==0.1.2
- ml-dtypes==0.4.1
- namex==0.0.8
- nltk==3.9.1
- numpy==2.0.2
- opt_einsum==3.4.0
- optree==0.13.1
- packaging==24.2
- pandas==2.2.3
- pillow==11.1.0
- protobuf==5.29.3
- Pygments==2.19.1
- pyparsing==3.2.1
- python-dateutil==2.9.0.post0
- pytz==2024.2
- regex==2024.11.6
- requests==2.32.3
- rich==13.9.4
- scikit-learn==1.6.1
- scipy==1.15.1
- six==1.17.0
- tensorboard==2.18.0
- tensorboard-data-server==0.7.2
- tensorflow==2.18.0
- tensorflow-io-gcs-filesystem==0.31.0
- tensorflow_intel==2.18.0
- termcolor==2.5.0
- threadpoolctl==3.5.0
- tqdm==4.67.1
- typing_extensions==4.12.2
- tzdata==2024.2
- urllib3==2.3.0
- Werkzeug==3.1.3
- wordcloud==1.9.4
- wrapt==1.17.1


you can easily run the following command to install all of them:
```bash
pip install -r requirements.txt
```
## 3. Run the following command to see the output:
process.py should be run first to process the data and save it in a csv file. Then, train.py file should be run to train models. You can simply run main.py file to do all of them at once.
```bash
python main.py
```
## 4. The dataset that is used in this project is the UCI ML Drug Review dataset. You can find more information about the dataset [here](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).