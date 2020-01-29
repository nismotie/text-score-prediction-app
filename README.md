# Command Line Review Score Predictor

## App Objective
This app is designed to answer the question "Can a command line app accurately predict the score out of five for a newly typed review of a given food stuff". The answer is "most of the time!". This app follows my [Data Science Bootcamp Capstone Project](https://github.com/nismotie/convolutional-sentiment-analysis), in which I trained a convolutional neural network to predict amazon review scores, so that it could then be used to predict previously unseen text. I saved the trained model and tokenizer from that project and used them to create a small command line app in which you enter an imagined or real piece of feedback for a foodstuff, and the score you think it should get, and the model will tell you what it think is the appropriate score based on the text alone. All of this information is recorded into a csv for the user's own purposes; I wanted to see how well the app performed during a science fair for the final project.

I have made this a separate repo because I have the lofty ambition of turning this command line app into a Flask and/or Django app, with a database to store the model information instead of a CSV. 

## Usage

This app relies on saved H5 models and a tokenizer both of which come from Keras, and so has the following requirements:
* Keras 2.3.0 or greater
* Python 3.6 or greater

I have used the same library from the original project in its entirety as it is used for a small amount of text preprocessing, and might be useful in other ways later in the app's development.

To get started simply run the predictory.py file from the command line:
```
python3 predictor.py
```
