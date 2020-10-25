import nltk
from collections import Counter
import numpy as np

def tokenize_data(text):
    """
    Takes a list of texts, tokenizes every text, and returns list of tokenized texts
    """
    text_tokens = []
    for utterance in text:
        text_tokens.append(nltk.tokenize.word_tokenize(utterance))
        
    return text_tokens


def get_frequent_keywords(texts, frequency_threshold):
    """
    Takes a list of texts and a frequency threshold
    Create and returns list of words above that threshold
    """
    frequent_keywords = []

    # Create a list of all words in the list of texts
    alltokens = []
    for text in texts:
        for token in text:
            alltokens.append(token)

    # Create a counter dictionary for all words
    kw_counter = Counter(alltokens)

    # Append words with count higher than threshold to frequent_keywords
    for word, count in kw_counter.items():
        if count>frequency_threshold:
            frequent_keywords.append(word)

    return frequent_keywords


def getAvgFeatureVecs(texts, keywords, stopwords, model, modelword_index, num_features):
    """
    Obtains the vector representation for every word and takes the average over all tokens for every text
    Returns a vector of average feature vectors for all texts
    """
    counter = 0

    # Initialise numpy vector with zeros of the type float32
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")
    
    # Iterate over all texts
    for text in texts:

        # Initialise numpy vector with zeros of the type float32
        featureVec = np.zeros(num_features,dtype="float32")
        nwords = 0
        
        # Iterate over all words
        for word in text:

            # Only include words that are frequent and exclude stopwords
            if word in keywords and not word in stopwords:         
                if word in modelword_index:
                    
                    # Take values from the model and normalize these values
                    featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))
                    nwords = nwords + 1
                else:
                    word = word.lower()
                    if word in modelword_index:
                        featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))
                        nwords = nwords + 1

        # Get average embedding vector for text and add to total list              
        textFeatureVecs[counter] = np.divide(featureVec, nwords)
        counter = counter+1
    
    # Turn infinitive values or NaN values to 0 scores
    textFeatureVecs = np.nan_to_num(textFeatureVecs) 
    
    return textFeatureVecs