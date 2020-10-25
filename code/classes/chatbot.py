import random
from code import EmotionClassifier, TopicClassifier
from code import get_keyword_dict

class Chatbot():
    """
    The Chatbot Class that ...
    """
    def __init__(self, responder, emotion_settings, emotion_models, topic_settings, embedding_model):
            self.responder = responder
            self.emotion_settings = emotion_settings
            self.emotion_models = emotion_models
            self.topic_settings = topic_settings
            self.embedding_model = embedding_model


    def respond(self, message):
        """
        Takes a message and generates a response by detecting an emotion with a specified
        emotion model and detecting keywords with a specified topic model.
        Returns the response, emotion, matching keywords and topic
        """
        # Get keyworddict with keywords mapped to every topic
        keyworddict = get_keyword_dict(self.responder)

        # Load emotion classifier and predict emotion
        emotion_classifier = EmotionClassifier(message, self.emotion_settings, self.emotion_models)
        emotion = emotion_classifier.predict()

        # Load topic classifier and predict topic and get matched words
        topic_classifier = TopicClassifier(message, keyworddict, self.topic_settings, self.embedding_model)
        topic, matched_words = topic_classifier.predict()

        # Generate response
        response = self.get_response(topic, emotion)

        return response, emotion, matched_words, topic


    def get_response(self, topiclabel, emotionlabel):
        """
        Takes a topic and an emotion and tries to find a response following the responder dictionary
        Returns the response
        """
        response = "I cannot respond to this"

        # Iterate over topics
        for topic in self.responder['topics']:
            # Iterate over emotion if topic matches
            if topic['topic'] == topiclabel:
                for emotion in topic['emotions']:
                    # Get random response from list of responses if emotion matches
                    if emotion['emotion'] == emotionlabel:
                        response = random.choice(emotion['responses'])

        return response