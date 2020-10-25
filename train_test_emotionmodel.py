import json
from code import EmotionModeltrainer, EmotionClassifier
from code import get_emotion_trainingdata, get_testdata, get_models, write_classification_report

# PROMPT TRAINING SETTINGS FROM USER

modelnr = input('What is the model number? (type number)\n')
teid = True if input('Do you want to extent the MELD with the TEID? (y/n)\n') == 'y' else False
classifier = input('Which classifier do you want to use? (naivebayes/svm)\n')
representations = input('What representations do you want to use? (embedding/bow/tfidf)\n')
frequency_threshold = int(input('What frequency threshold do you want to use? (type number)\n'))
if representations == 'embedding':
    embedding_model = input('What embedding model do you want to use? (glove-wiki-gigaword-300/glove-twitter-200/word2vec-google-news-300)\n')
else:
    embedding_model = None
if embedding_model == 'glove-twitter-200':
    dimensions = 200
else:
    dimensions = 300
stopwords = True if input('Do you want to exclude stopwords? (y/n)\n') == 'y' else False
balanced_data = True if input('Do you want to balance (over-sampling) the training data? (y/n)\n') == 'y' else False

# WRITE SETTINGS

settings = {'modelnr': modelnr, 'teid': teid, 'classifier': classifier, 'representations': representations, \
    'frequency_threshold': frequency_threshold, 'embedding_model': embedding_model, 'dimensions': dimensions, \
        'stopwords': stopwords, 'balance_data': balanced_data}

models_path = f"./models/emotion/{modelnr}"
settings_path = f"{models_path}/settings_{modelnr}.json"

with open(settings_path, 'w') as outfile:
    json.dump(settings, outfile)

# TRAIN MODEL

print('Training model...')
training_texts, training_labels = get_emotion_trainingdata(MELD_path='./data/MELD/train_sent_emo.csv', TEID=teid, TEID_path='./data/TEID')
modeltrainer = EmotionModeltrainer(training_texts, training_labels, settings, models_path)
modeltrainer.run()

# GET TEST DATA

texts, labels, topics = get_testdata(filepath='./data/test_set.csv')

# OPTION TO GET BASELINE CLASSIFICATION REPORT

if input("Do you want to generate a baseline classification report before testing a model? (y/n)\n") == 'y':
    baseline_result = ['neutral' for i in range(len(labels))]
    write_classification_report(labels, baseline_result, './models/emotion/baseline_classification_report.csv')

# TEST MODEL

print('Testing model ...')
models = get_models(models_path, settings)
predictions = []

for text in texts:
    emotion_classifier = EmotionClassifier(text, settings, models)
    prediction = emotion_classifier.predict()
    predictions.append(prediction)

write_classification_report(labels, predictions, f'{models_path}/classification_report.csv')
print(f"Classification report is saved in '{models_path}/classification_report.csv'!")
