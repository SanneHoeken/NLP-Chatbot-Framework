import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def filter_tokens(tokens):
    """
    Takes a list of tokens, filters stopwords and returns filtered list
    """
    stopwords = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token.lower() in stopwords]

    return filtered_tokens


def process_text(text, processing):
    """
    Takes a text and returns the text tokenized/lemmatized
    """
    # Tokenize text 
    tokens = nltk.tokenize.word_tokenize(text)

    # Lemmatize text if specified
    if processing == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmas

    return tokens


def similar_words_from_wordnet(token, hyponyms, hypernyms):
    """
    Takes a token and tries to find similar words in Wordnet
    Returns list with synonyms and hyponyms and hypernyms if specified
    """
    similar_synsets = set()

    # Checks if token is in Wordnet vocabulary
    if wordnet.synsets(token):
        
        # Iterate over token's synsets
        for synset in wordnet.synsets(token):
            
            # Add synset to list of similar synsets
            similar_synsets.add(synset)

            # Get and add hyponym synsets if specified
            if hyponyms:
                for hyponym in synset.hyponyms():
                    similar_synsets.add(hyponym)

            # Get and add hyponym synsets if specified
            if hypernyms:
                for hypernym in synset.hypernyms():
                    similar_synsets.add(hypernym)
    
    # Get list of lemmas of all synsets
    similar_lemmas = get_synsets_lemmas(similar_synsets)

    return similar_lemmas

def get_synsets_lemmas(synsets):
    """
    Take a list of synsets and returns a list of lemmas of all synsets
    """
    lemmas = []

    for synset in synsets:
        for lemma in synset.lemmas():
            word = lemma.name()
            lemmas.append(word)

    return lemmas

def topn_similar_words(token, similar_words, topn, measure, embedding_model):
    """
    Takes a token and a list of similar words ranks the similar words 
    based on a specified similarity measure
    Returns a specified topn similar words from this ranking
    """
    similarity_dict = dict()

    # Get similarity for all words and map to word in dict
    for word in similar_words:
        similarity = get_similarity(token, word, measure, embedding_model)
        if similarity:
            similarity_dict[word] = similarity
    
    # Sort dict keys by its values
    sorted_words = sorted(similarity_dict, key=similarity_dict.get, reverse=True)

    return sorted_words[:topn]


def get_similarity(token1, token2, measure, embedding_model):
    """
    Calculates similarity between two given tokens based on specified measuremnent
    """
    similarity = None

    # Calculates similarity based on embedding model measurements
    if measure == 'embedding_model':
        if token1 in embedding_model.vocab and token2 in embedding_model.vocab:
            similarity = embedding_model.similarity(token1, token2)

    # Calculates similarity based on Wordnet measurements
    elif any([measure == 'path', measure == 'lch', measure == 'wup']):
        
        similarities = []

        # Get for both tokens their Wordnet synsets
        # Calculate similarity for every synset combination and append to list
        for synset1 in wordnet.synsets(token1):
            for synset2 in wordnet.synsets(token2):
                similarity = calculate_synset_similarity(measure, synset1, synset2)
                if similarity:
                    similarities.append(similarity)
        
        # Get the maximum similarity measure
        if similarities:
            similarity = max(similarities) 

    return similarity


def calculate_synset_similarity(measure,s1, s2):
    """
    Calculates and returns the similarity between two synsets based on specified measurement
    """
    if measure == 'path':
        similarity = s1.path_similarity(s2)
    elif measure == 'lch':
        similarity = s1.lch_similarity(s2) 
    elif measure == 'wup':
        similarity = s1.wup_similarity(s2)

    return similarity 


