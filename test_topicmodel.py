from code import TopicClassifier
from code import get_testdata, get_keyword_dict, write_classification_report
import gensim.downloader as api
import json

# GET TEST DATA

texts, emotions, labels = get_testdata(filepath='./data/test_set.csv')

# OPTION TO GET BASELINE CLASSIFICATION REPORT

if input("Do you want to generate a baseline classification report before testing a model? (y/n)\n") == 'y':
    baseline_result = ['people' for i in range(len(labels))]
    write_classification_report(labels, baseline_result, './models/topic/baseline_classification_report.csv')

# PROMPT MODEL SETTINGS FROM USER

modelnr = int(input('What is the model number? (type number)\n'))
model = input('What semantic model do you want to use? (embedding/wordnet/embedding+wordnet)\n')
filter_text = True if input('Do you want to exclude stopwords? (y/n)\n') == 'y' else False
neighborhood = int(input('What is the size of the word neighborhood you want to use? (type number)\n'))
processing = input('What kind of text processing do you want to use? (tokenize/lemmatize)\n')
if not model == 'embedding':
    hyponyms = True if input('Do you also want to involve hyponymic relationships in the formation of a word neighborhood? (y/n)\n') == 'y' else False
    hypernyms = True if input('Do you also want to involve hypernymic relationships in the formation of a word neighborhood? (y/n)\n') == 'y' else False
else:
    hyponyms = None
    hypernyms = None
measure = input('Which similarity measurement do you want to use? (embedding_model/path/lch/wup)\n')
threshold = float(input('What is the similarity threshold on which the matching with keywords is based? (type number between 0 and 1)\n'))
if measure == 'embedding_model' or not model == 'wordnet':
    embedding_model = input('What embedding model do you want to use? (glove-wiki-gigaword-300/glove-twitter-200/word2vec-google-news-300)\n')
else:
    embedding_model = None

# WRITE SETTINGS

settings = {'modelnr': modelnr, 'model': model, 'embedding_model': embedding_model, 'filter_text': filter_text, 'neighborhood': neighborhood,\
     'processing': processing, 'hyponyms': hyponyms, 'hypernyms': hypernyms, 'measure': measure, 'threshold': threshold}

model_path = f"./models/topic/{modelnr}"
settings_path = f"{model_path}/settings_{modelnr}.json"
with open(settings_path, 'w') as outfile:
    json.dump(settings, outfile)

# LOAD KEYWORDDICT

with open('./data/responder.json') as infile:
    responder = json.load(infile)
keyworddict = get_keyword_dict(responder)

# TEST MODEL

print('Testing model ...')
embedding_model = api.load(settings['embedding_model'])
predictions = []

for text in texts:
    topic_classifier = TopicClassifier(text, keyworddict, settings, embedding_model)
    prediction = topic_classifier.predict()
    predictions.append(prediction[0])

write_classification_report(labels, predictions, f'{model_path}/classification_report.csv')
print(f"Classification report is saved in '{model_path}/classification_report.csv'!")