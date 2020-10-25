from collections import defaultdict
from code import filter_tokens, process_text, similar_words_from_wordnet, topn_similar_words, get_similarity

class TopicClassifier():
    """
    The TopicClassifier Class that predicts the topic of a text with a specified topic classification model
    """
    def __init__(self, text, keyworddict, settings, embedding_model):
        self.text = text
        self.keyworddict = keyworddict
        self.settings = settings
        self.embedding_model = embedding_model
        self.similar_words = defaultdict(set)
    

    def predict(self):
        """
        Predicts the topic label of a text by enriching the text with similar words
        and than compares these enriched text with predefined keywords in a way that 
        meets all specified settings. Returns the prediction.
        Returns the topic label with the most matches along with the matched words.
        """
        self.set_similar_words()

        prediction = 'other'
        predictions_matched_words = set()
        most_matches = 0

        # Iterate over topics in keyworddict
        for topic in self.keyworddict:
            
            # Get matches with keywords
            keywords = self.keyworddict[topic]
            matched_words = self.get_matched_words(keywords)
            
            # Keep topic label and matched words if most matches so far    
            n_matches = len(matched_words)
            if n_matches > most_matches:
                most_matches = n_matches
                prediction = topic
                predictions_matched_words = matched_words

        return prediction, predictions_matched_words     
        

    def set_similar_words(self):
        """
        Sets the similar_words dictionary by finding similar words for every token
        Put every similar word as key in the dictionary mapped to the token
        """
        # Process (tokenize/lemmatize) the text
        tokens = process_text(self.text, self.settings['processing'])
        
        # Exclude stopwords from text if specified in settings
        if self.settings['filter_text']:
            tokens = filter_tokens(tokens)

        # Iterate over tokens in text    
        for token in set(tokens):

            # Add the token itself to dict
            self.similar_words[token].add(token)

            # Get similar words to token and add them to dictionary mapped to token
            similar_to_token = self.get_similar_to_token(token)
            for word in similar_to_token:
                self.similar_words[word].add(token)


    def get_similar_to_token(self, token):
        """
        Takes a token and gets a specified top n (= neighborhood size) similar words 
        based on a specified model and returns the set of similar words
        """
        similar_to_token = set()
        neighborhood = self.settings['neighborhood']

        # Divides neighborhood size by two if both an embedding model and Wordnet should be used
        if self.settings['model'] == 'embedding+wordnet':
            neighborhood = int(neighborhood/2)
        
        # Get top n similar words from specified embedding model and add to set
        if self.settings['model'] == 'embedding' or self.settings['model'] == 'embedding+wordnet':
            if token in self.embedding_model.vocab:
                word_neighborhood = self.embedding_model.most_similar(positive=[token], topn=neighborhood)
                for item in word_neighborhood:
                    word = item[0].lower()
                    similar_to_token.add(word)

        # Get top n similar words from Wordnet if specified and add to set
        if self.settings['model'] == 'wordnet' or self.settings['model'] == 'embedding+wordnet':
            similar_words = similar_words_from_wordnet(token, self.settings['hyponyms'], self.settings['hypernyms'])
            topn_words = topn_similar_words(token, similar_words, neighborhood,\
                 self.settings['measure'], self.embedding_model)
            similar_to_token.update(topn_words)
        
        return similar_to_token
    
    
    def get_matched_words(self, keywords):
        """
        Takes a set of keywords and tries to match these words with the similar words,
        only accepts match if similarity measure is above specified threshold.
        Puts keywords as key in dict mapped to tokens in original text that are
        similar to the words that matched these keywors. Returns the dict.
        """
        matched_words = defaultdict(set)

        # Iterate over keywords and similar words
        for keyword in keywords:
            for word in self.similar_words.keys():

                # Get similarity between every combination
                similarity = get_similarity(keyword, word, self.settings['measure'], self.embedding_model)
                
                # Adds keyword to dict mapped to text tokens similar to matched words
                # only if similarity is above threshold specified in settings
                if similarity:
                    if similarity >= self.settings['threshold']:
                        matched_words[keyword].update(self.similar_words[word])

        return matched_words
