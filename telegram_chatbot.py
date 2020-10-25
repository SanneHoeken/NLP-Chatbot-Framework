from code import BotHandler, Chatbot
from code import get_models, get_settings
import gensim.downloader as api
import json

# Load bot token
with open('./data/bot_token.json') as infile:
    content = json.load(infile)

# Set token, user id and bot
token = content['bot_token']
user_id = 990148273
bot = BotHandler(token)

# Load responder json-file
with open('./data/responder.json') as infile:
    responder = json.load(infile)

# Prompt emotion and topic model from user
emotion_modelnr = input("Emotion model number: ")
topic_modelnr = input("Topic model number: ")

# Get emotion settings and models
emotion_model_path = f"./models/emotion/{emotion_modelnr}"
emotion_settings_path = f"{emotion_model_path}/settings_{emotion_modelnr}.json"
emotion_settings = get_settings(emotion_settings_path)
emotion_models = get_models(emotion_model_path, emotion_settings)

# Get topic model settings and embedding model
topic_settings_path = f"./models/topic/{topic_modelnr}/settings_{topic_modelnr}.json"
topic_settings = get_settings(topic_settings_path)

embedding_model = None
if topic_settings['embedding_model']:
    if emotion_settings['embedding_model'] == topic_settings['embedding_model']:
        embedding_model = emotion_models['embedding_model']
    else: 
        embedding_model = api.load(topic_settings['embedding_model'])

# Get last message sent to bot
last_message = bot.get_last_message_by(user_id)

# Set chatbot and generate response to last message
chatbot = Chatbot(responder, emotion_settings, emotion_models, topic_settings, embedding_model)
response, emotion, matched_words, topic = chatbot.respond(last_message)

# Send response
bot.send_message_to(user_id, response)

# Print results
print(f"Last message: {last_message}")
print(f"Emotion detected: {emotion}")
print(f"Keywords detected: {matched_words}")
print(f"Topic detected: {topic}")
print(f"Response: {response}")
