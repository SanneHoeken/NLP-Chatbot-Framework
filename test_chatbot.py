from code import Chatbot
from code import get_models, get_settings, get_testdata
import gensim.downloader as api
import json, csv
from collections import defaultdict

# GET TEST DATA
texts, emotions, labels = get_testdata(filepath='./data/test_set.csv')

# LOAD RESPONDER JSON-FILE
with open('./data/responder.json') as infile:
    responder = json.load(infile)

# TEST FOUR DIFFERENT CHATBOTS
# four combinations of the two emotion models 4 and 5 and the two topic models 1 and 4

test_results = defaultdict(list)
test_results['texts'] = texts

for emotion_modelnr in ['5', '4']:

    # Get emotion settings and models
    emotion_model_path = f"./models/emotion/{emotion_modelnr}"
    emotion_settings_path = f"{emotion_model_path}/settings_{emotion_modelnr}.json"
    emotion_settings = get_settings(emotion_settings_path)
    emotion_models = get_models(emotion_model_path, emotion_settings)
    
    for topic_modelnr in ['1', '4']:

        setup = (emotion_modelnr, topic_modelnr)

        # Get topic model settings and embedding model
        topic_settings_path = f"./models/topic/{topic_modelnr}/settings_{topic_modelnr}.json"
        topic_settings = get_settings(topic_settings_path)

        embedding_model = None
        if topic_settings['embedding_model']:
            if emotion_settings['embedding_model'] == topic_settings['embedding_model']:
                embedding_model = emotion_models['embedding_model']
            else: 
                embedding_model = api.load(topic_settings['embedding_model'])

        # Set chatbot
        chatbot = Chatbot(responder, emotion_settings, emotion_models, topic_settings, embedding_model)

        # Generate responses to test data texts
        for text in texts:
            response, emotion, matched_words, topic = chatbot.respond(text)
            test_results[setup].append(response)

# WRITE RESULTS TO CSV

with open("test_results.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(list(test_results.keys()))
   writer.writerows(zip(*list(test_results.values())))

        

        
