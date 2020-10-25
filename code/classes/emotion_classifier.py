from code import tokenize_data, getAvgFeatureVecs
import nltk

class EmotionClassifier():
    """
    The EmotionClassifier Class that predicts the emotion of a text with a specified emotion classification model
    """
    def __init__(self, text, settings, models):
            self.text = [text]
            self.models = models
            self.settings = settings
            

    def predict(self):
        """
        Predicts the emotion label of a text using an emotion classification model that meets all specified settings 
        """
        representation = self.get_representation()
        classifier = self.models['classifier']
        prediction = classifier.predict(representation)
        label = self.decode_label(prediction[0])

        return label


    def get_representation(self):
        """
        Returns representation as specified using the loaded models
        """
        representation = None

        # Declare stopwords to exclude if specified
        stopwords = nltk.corpus.stopwords.words('english') if self.settings['stopwords'] == True else None

        # Create Bag-of-Words vector representations if specified
        if self.settings['representations'] == 'bow':
            
            # Transform the text into count vectors using the loaded the vectorizer
            vectorizer = self.models['vectorizer']
            representation = vectorizer.transform(self.text)

        # Create information value vector representations if specified
        elif self.settings['representations'] == 'tfidf':
            
            # Transform the text into count vectors using the loaded the vectorizer
            vectorizer = self.models['vectorizer']
            representation = vectorizer.transform(self.text)
            
            # Transform counts into information value scores using the loaded transformer
            tfidf_transformer = self.models['tfidf_transformer']
            representation = tfidf_transformer.transform(representation)

        # Create embedding representations if specified
        elif self.settings['representations'] == 'embedding':

            # Tokenize the text
            self.text = tokenize_data(self.text)

            # Load all the models needed for creating embeddings
            frequent_keywords = self.models['frequent_keywords']
            embedding_model = self.models['embedding_model']
            index2word_set = self.models['index2word_set']

            # Extract the embedding representation
            representation = getAvgFeatureVecs(self.text, frequent_keywords, stopwords,\
                 embedding_model, index2word_set, self.settings['dimensions'])
        
        return representation
    

    def decode_label(self, prediction):
        """
        Turns the given numerical label into string value using the loaded encoder and returns it
        """
        le = self.models['label_encoder']
        label = le.classes_[prediction] 
        return label