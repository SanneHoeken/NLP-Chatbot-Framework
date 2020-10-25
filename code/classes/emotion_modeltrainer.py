import nltk, pickle, pandas, json
import gensim.downloader as api
from sklearn import preprocessing, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from code import tokenize_data, get_frequent_keywords, getAvgFeatureVecs

class EmotionModeltrainer():
    """
    The EmotionModeltrainer Class that trains an emotion classification model
    """
    def __init__(self, texts, labels, settings, models_path):
        self.texts = texts
        self.labels = labels
        self.settings = settings
        self.models_path = models_path
        self.models = dict()

    def run(self):
        """
        Trains an emotion classification model that meets all specified settings
        Saves all (preprocessing) models to disk 
        """
        representations = self.get_representations()
        self.encode_labels()
        if self.settings['balance_data']:
            representations = self.resample_data(representations)
        self.train_model(representations)
        self.save_models()

    def get_representations(self):
        """
        Creates and returns representations as specified and stores the models used for creation
        """
        representations = None

        # Declare stopwords to exclude if specified
        stopwords = nltk.corpus.stopwords.words('english') if self.settings['stopwords'] == True else None

        # Create Bag-of-Words vector representations if specified
        if self.settings['representations'] == 'bow':
            
            # Transform the text into count vectors and store the vectorizer
            vectorizer = CountVectorizer(min_df=self.settings['frequency_threshold'],\
                 tokenizer=nltk.word_tokenize, stop_words=stopwords)
            representations = vectorizer.fit_transform(self.texts)
            self.models['vectorizer'] = vectorizer

        # Create information value vector representations if specified
        elif self.settings['representations'] == 'tfidf':
            
            # Transform the text into count vectors and store the vectorizer
            vectorizer = CountVectorizer(min_df=self.settings['frequency_threshold'],\
                 tokenizer=nltk.word_tokenize, stop_words=stopwords)
            representations = vectorizer.fit_transform(self.texts)
            self.models['vectorizer'] = vectorizer
            
            # Transform counts into information value scores and store the transformer
            tfidf_transformer = TfidfTransformer()
            representations = tfidf_transformer.fit_transform(representations)
            self.models['tfidf_transformer'] = tfidf_transformer

        # Create embedding representations if specified
        elif self.settings['representations'] == 'embedding':

            # Tokenize the texts and labels
            self.texts = tokenize_data(self.texts)
            self.labels = tokenize_data(self.labels)

            # Create and store list of words above the preset frequency threshold
            frequent_keywords = get_frequent_keywords(self.texts, self.settings['frequency_threshold'])   
            self.models['frequent_keywords'] = frequent_keywords
            
            # Load and store pre-trained embeddings according to specified model
            embedding_model = api.load(self.settings['embedding_model'])
            self.models['embedding_model'] = embedding_model
            
            # Extract and store for each text the embedding representation
            index2word_set = set(embedding_model.wv.index2word)
            self.models['index2word_set'] = index2word_set
            representations = getAvgFeatureVecs(self.texts, frequent_keywords, stopwords,\
                 embedding_model, index2word_set, self.settings['dimensions'])
        
        return representations

    def encode_labels(self):
        """
        Turns labels into numerical values ands stores the encoder
        """
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        self.models['label_encoder'] = le

    def resample_data(self, representations):
        """
        Balances the data using over-sampling and returns the resampled representations
        source: https://medium.com/@satishkorapati/dealing-with-imbalanced-dataset-for-multi-class-text-classification-having-multiple-categorical-2a43fc8de009
        """
        smote = SMOTE()
        resampled_representations, self.labels = smote.fit_sample(representations, self.labels)

        return resampled_representations

    def train_model(self, representations):
        """
        Trains the classifier that is specified and stores it
        """
        # Train and store a Naive Bayes classifier if specified 
        if self.settings['classifier'] == 'naivebayes':
            naive_bayes = MultinomialNB().fit(representations, self.labels)
            self.models['classifier'] = naive_bayes
        
        # Train and store a Linear Support Vector Machine classifier if specified
        elif self.settings['classifier'] == 'svm':
            svm_linear = svm.LinearSVC(max_iter=2000)
            svm_linear.fit(representations, self.labels)
            self.models['classifier'] = svm_linear

    def save_models(self):
        """
        Saves all stored models to disk
        """
        for model in self.models:
            filename = f'{self.models_path}/{model}.sav'
            pickle.dump(self.models[model], open(filename, 'wb'))
            
