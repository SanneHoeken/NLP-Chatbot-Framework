import glob, json, os, pickle, sys
import pandas as pd
import gensim.downloader as api
from sklearn.metrics import classification_report

def get_testdata(filepath):
    """
    Takes path to test data 
    Opens file and extracts texts, topic and emotion labels
    Returns texts, topics and emotions as pandas Series
    """
    dftest = pd.read_csv(filepath, sep=';')
    texts = dftest['text']
    emotions = dftest['emotion']
    topics = dftest['topic']

    return texts, emotions, topics

def get_emotion_testdata(filepath):
    """
    Takes path to test data 
    Opens file and extracts texts and emotion labels
    Returns two pandas Series containing the texts and the labels
    """
    dftest = pd.read_csv(filepath, sep=';')
    texts = dftest['text']
    labels = dftest['emotion']

    return texts, labels


def get_topic_testdata(filepath):
    """
    Takes path to test data 
    Opens file and extracts texts and topic labels
    Returns two pandas Series containing the texts and the labels
    """
    dftest = pd.read_csv(filepath, sep=';')
    texts = dftest['text']
    labels = dftest['topic']

    return texts, labels


def get_keyword_dict(responder):
    """
    Takes a responder dictionary and returns a dictionary with keywords mapped to topics 
    """
    keyworddict = dict()    
    for topic in responder['topics']:
        keyworddict[topic['topic']] = topic['keywords']

    return keyworddict


def get_models(models_path, settings):
    """
    Loads and returns all the models from model path and adds embedding model if necessary
    """
    models = dict()

    # Load all models that are present in model path to dictionary
    for filename in glob.glob(f'{models_path}/*.sav'):
        model = pickle.load(open(filename, 'rb'))
        models[os.path.basename(filename)[:-4]] = model
    
    # Loads and add embedding model if necessary
    if not 'embedding_model' in models and settings['embedding_model']:
        models['embedding_model'] = api.load(settings['embedding_model'])

    return models


def get_settings(settings_path):
    """
    Opens the json-file with settings and loads and returns them
    """
    with open(settings_path) as infile:
        settings = json.load(infile)
    
    return settings


def write_classification_report(gold, predictions, outfile_path):
    """
    Generates classification report for given gold labels and predictions
    Writes the report as csv-file to specified filepath
    """
    report = classification_report(gold, predictions, digits = 7, output_dict = True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(outfile_path)


def get_emotion_trainingdata(MELD_path, TEID=False, TEID_path=None):
    """
    Takes path to the MELD and path to the TEID if TEID is set to True,
    Opens file(s), deals with encoding and extracts texts and emotion labels.
    Returns two pandas Series containing the texts and the labels
    """
    if not os.path.exists(MELD_path) or (TEID and not os.path.exists(TEID_path)):
        print('Path to training data does not exist.')
        sys.exit()

    # Read the MELD csv-file into DataFrame 
    dftrain = pd.read_csv(MELD_path)

    # Try to solve encoding issues
    dftrain['Utterance'] = dftrain['Utterance'].str.replace("\x92|\x97|\x91|\x93|\x94|\x85", "'")

    # Extract texts and emotion labels
    texts = dftrain['Utterance']
    labels = dftrain['Emotion']

    # Get the TEID if specified
    if TEID: 

        # Iterate over files in TEID directory
        for filename in glob.glob(f'{TEID_path}/*.csv'):
            dftrain = pd.read_csv(filename, sep=';')

            # Only include data if emotion intensity is higher than 0.5
            dftrain = dftrain.loc[dftrain['intensity'] > 0.5]
                
            # Concatenate the TEID data to the MELD data  
            texts = texts.append(dftrain['text'])
            labels = labels.append(dftrain['emotion'])  
    
    return texts, labels