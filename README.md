# NLP-Chatbot-Framework

This project is part of the master course Introduction to Human Language Technology, which is part of the Humanities Research master at VU University Amsterdam.
October 2020.

### Project

In this project I have set up a framework in which a chatbot can be built that can respond to the emotion and topic of an utterance. In the building process, the chatbot is provided with a classification model that can detect emotions and a classification model that can detect topics. The latter model uses a set of keywords that are linked to a topic. These keywords are included in a responder file. Using the same responder file, the chatbot can return an associated response based on the detected emotion and topic. In the framework that I have set up, different classification models can easily be implemented to see what the consequences are for the performance of the chatbot. Both the emotion and the topic classification models include many parts that offer many variation possibilities. For this project I selected a set of settings and for each setting a set of variation options, which are explained in the report pdf-file. I set up the framework in such a way that you can easily adjust these settings. 

## Getting started

### Requirements

This codebase is written entirely in Python 3.7. requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:

```
pip install -r requirements.txt
```

Or via conda:

```
conda install --file requirements.txt
```

### Structure

The following list describes the most important folders and files in the project, and where to find them:

- **/code**: contains all code of this project
  - **/code/classes**: contains the necessary classes for this project
  - **/code/utils**: contains the helper functions for this project
- **/data**: contains the training and test data, the responder file and the bot token
- **/models**: contains all built models in this project
  - **/models/emotion**: contains all built emotion classification models
  - **/models/topic**: contains all built topic classification models

## Using the main programs

1. **Training and testing of an emotion classification model.**
  This program can be run by calling:
  ``
  python train_test_emotionmodel.py
  ``
  The user-friendly program will ask you to enter the desired settings. After the execution of the program, the model with a classification report and settings are saved under models.
2. **Setting up and testing a topic classification model.**
  This program can be run by calling:
  ``
  python test_topicmodel.py
  ``
  The user-friendly program will ask you to enter the desired settings. After the execution of the program, the model with a classification report and the settings are saved under models.
3. **Testing four chatbot systems.**
  This program can be run by calling:
  ``
  python test_chatbot.py
  ``
  The program tests four different chatbot systems and writes the results to a csv-file.
4. **Running a chatbot system via Telegram.**
  This program can be run by calling:
  ``
  python test_chatbot.py
  ``
  The program instantiates a bot that can respond to messages from a user via Telegram (default user id = 990148273). The program will ask you to enter model numbers of the classification models on the basis of which the system generates a response.

## Author
- Sanne Hoeken (student number: 2710599)
