This README file provides information about the Music Genre Classifier project.

## Features:

Trains a machine learning model to identify music genre based on features extracted from audio files. (Note: Requires a separate audio feature extraction library)
Classifies the genre of new music files provided by the user.
Offers options to train the model on a provided dataset or load a pre-trained model.

## Installation:

Clone this repository using git clone https://github.com/Alexstandall179/music-genre-classifier.git (replace <username> with your GitHub username).
Open a terminal and navigate to the project directory using cd music-genre-classifier.
Install the required Python libraries using pip install [package for audio feature extraction] scikit-learn (Replace [package for audio feature extraction] with the actual library for audio feature extraction like librosa).

## Usage:

Run the script using python music_genre_classifier.py.
The script will prompt you to choose between training a new model or loading a pre-trained model.
If training a new model:
Specify the path to the dataset containing music files labeled with their genres.
If loading a pre-trained model:
Specify the path to the saved model file.
(Optional - for training a new model) Define hyperparameters for the machine learning model (e.g., number of training epochs).
After training (or loading the model), you can provide the path to a music file for genre classification.
The script will extract features from the audio file, use the trained model to predict the genre, and display the predicted genre.
