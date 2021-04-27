#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/SETIADEEPANSHU/Audio_Classification_Using_CNN/blob/master/Notebooks/Excercise_Audio_Classification_Using_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Excercise

# 

# # Overview
# 
# Sounds are all around us. Whether directly or indirectly, we are always in contact with audio data. Sounds outline the context of our daily activities, ranging from the conversations we have when interacting with people, the music we listen to, and all the other environmental sounds that we hear on a daily basis such as a car driving past, the patter of rain, or any other kind of background noise. The human brain is continuously processing and understanding this audio data, either consciously or subconsciously, giving us information about the environment around us.
# 
# Automatic environmental sound classification is a growing area of research with numerous real world applications. Whilst there is a large body of research in related audio fields such as speech and music, work on the classification of environmental sounds is comparatively scarce. Likewise, observing the recent advancements in the field of image classification where convolutional neural networks are used to to classify images with high accuracy and at scale, it begs the question of the applicability of these techniques in other domains, such as sound classification, where discrete sounds happen over time.
# 
# 

# # Problem Statement
# 
# The goal of this project, is to apply Deep Learning techniques to the `classification of environmental sounds` , specifically focusing on the identification of particular urban sounds. 
# 
# When given an audio sample in a computer readable format (such as a .wav file) of a few seconds duration, we want to be able to determine if it contains one of the target urban sounds with a corresponding likelihood score.
# 
# ![alt text](https://i.ibb.co/kBBVLX4/image.png)

# # Motivation
# 
# There is a plethora of real world applications for this research, such as:
# 
# *   Content-based multimedia indexing and retrieval
# *   Assisting deaf individuals in their daily activities
# *   Smart home use cases such as 360-degree safety and security capabilities
# *   Automotive where recognising sounds both inside and outside of the car can improve safety
# *  Industrial uses such as predictive maintenance
# * Heart, snoring sound classification
# 
# 
# 
# 

# # Outline
# 
# 
# 1.   Data Exploration and Visualisation
# 2.   Data Preprocessing and Data Splitting
# 3.   Model Training and Evaluation
# 4.   Model Refinement
# 

# ## PART 1. Data Exploration and Visualisation

# ### UrbanSound dataset
# 
# For this project we will use a dataset called Urbansound8K. The dataset contains 8732 sound excerpts (<=4s) of urban sounds from 10 classes, which are:
# 
# - Air Conditioner
# - Car Horn
# - Children Playing
# - Dog bark
# - Drilling
# - Engine Idling
# - Gun Shot
# - Jackhammer
# - Siren
# - Street Music
# 
# The accompanying metadata contains a unique ID for each sound excerpt along with it's given class name.
# 
# A sample of this dataset is included with the accompanying git repo and the full dataset can be downloaded from [here](https://urbansounddataset.weebly.com/urbansound8k.html).

# ### Audio sample file data overview
# 
# These sound excerpts are digital audio files in .wav format. 
# 
# Sound waves are digitised by sampling them at discrete intervals known as the sampling rate (typically 44.1kHz for CD quality audio meaning samples are taken 44,100 times per second). 
# 
# Each sample is the amplitude of the wave at a particular time interval, where the bit depth determines how detailed the sample will be also known as the dynamic range of the signal (typically 16bit which means a sample can range from 65,536 amplitude values). 
# 
# This can be represented with the following image: 
# <img src="https://i.imgur.com/PJeiFdy.png">
# 
# Therefore, the data we will be analysing for each sound excerpts is essentially a one dimensional array or vector of amplitude values. 

# ### Analysing audio data 
# 
# For audio analysis, we will be using the following libraries: 
# 
# #### 1. IPython.display.Audio 
# 
# This allows us to play audio directly in the Jupyter Notebook. 
# 
# #### 2. Librosa 
# 
# librosa is a Python package for music and audio processing by Brian McFee and will allow us to load audio in our notebook as a numpy array for analysis and manipulation. 
# 
# You may need to install librosa using pip as follows: 
# 
# `pip install librosa` 

# In[ ]:


# Install librosa - Music & Audio processing library
get_ipython().system('pip install librosa')


# Initial Setup 

# In[ ]:


# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Change the directory
import os
os.chdir("/content/drive/My Drive/")


# In[ ]:


# Clone the github repository
get_ipython().system('git clone https://github.com/SETIADEEPANSHU/Audio_Classification_Using_CNN.git')


# In[ ]:


# Change the directory
import os
os.chdir("Audio_Classification_Using_CNN/")


# In[ ]:


# Get the latest updates from the cloned github repository
# !git pull


# ### Auditory inspection 
# 
# We will use `IPython.display.Audio` to play the audio files so we can inspect aurally. 

# In[ ]:


import IPython.display as ipd

ipd.Audio('UrbanSound Dataset sample/audio/100032-3-0-0.wav')


# ### Visual inspection
# 
# We will load a sample from each class and visually inspect the data for any patterns. We will use librosa to load the audio file into an array then use librosa.display and matplotlib to display the waveform. 

# In[ ]:


# Load imports

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# In[ ]:


# Class: Air Conditioner

filename = 'UrbanSound Dataset sample/audio/100852-0-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# Its your turn now to run and visually inspect!

# In[ ]:


# Class: Car horn 

filename = 'UrbanSound Dataset sample/audio/100648-1-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Children playing 

filename = 'UrbanSound Dataset sample/audio/100263-2-0-117.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Dog bark

filename = 'UrbanSound Dataset sample/audio/100032-3-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Drilling

filename = 'UrbanSound Dataset sample/audio/103199-4-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Engine Idling 

filename = 'UrbanSound Dataset sample/audio/102857-5-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Gunshot

filename = 'UrbanSound Dataset sample/audio/102305-6-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Jackhammer

filename = 'UrbanSound Dataset sample/audio/103074-7-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Siren

filename = 'UrbanSound Dataset sample/audio/102853-8-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:


# Class: Street music

filename = 'UrbanSound Dataset sample/audio/101848-9-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# #### Observations 
# 
# 
# From a visual inspection we can see that it is tricky to visualise the difference between some of the classes. 
# 
# Particularly, the waveforms for reptitive sounds for air conditioner, drilling, engine idling and jackhammer are similar in shape.  
# 
# Likewise the peak in the dog barking sample is simmilar in shape to the gun shot sample (albeit the samples differ in that there are two peaks for two gunshots compared to the one peak for one dog bark). Also, the car horn is similar too. 
# 
# There are also similarities between the children playing and street music. 
# 
# The human ear can naturally detect the difference between the harmonics, it will be interesting to see how well a deep learning model will be able to extract the necessary features to distinguish between these classes. 
# 
# 
# However, it is easy to differentiate from the waveform shape, the difference between certain classes such as dog barking and jackhammer. 

# ### Grabbing Full Dataset now

# In[ ]:


# Download the full dataset
get_ipython().system('wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz')


# In[ ]:


# Unzip the Dataset
get_ipython().system('tar -xvf UrbanSound8K.tar.gz')


# ### Dataset Metadata 
# 
# Here we will load the UrbanSound metadata .csv file into a Panda dataframe. 

# In[ ]:


import pandas as pd
metadata = pd.read_csv('UrbanSound Dataset sample/metadata/UrbanSound8K.csv')
metadata.head()


# ### Class distributions

# In[ ]:


print(metadata.class_name.value_counts())


# The total count is `8732`

# #### Observations 
# 
# Here we can see the Class labels are unbalanced. Although 7 out of the 10 classes all have exactly 1000 samples, and siren is not far off with 929, the remaining two (car_horn, gun_shot) have significantly less samples at 43% and 37% respectively. 
# 
# This will be a concern and something we may need to address later on. 

# ### Audio sample file properties
# 
# Next we will iterate through each of the audio sample files and extract number of audio channels, sample rate and bit-depth. 

# In[ ]:


# Load various imports 
import pandas as pd
import os
import librosa
import librosa.display

from Notebooks.helpers.wavfilehelper import WavFileHelper
wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath('UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])


# #### Audio channels 
# 
# Most of the samples have two audio channels (meaning stereo) with a few with just the one channel (mono). 
# 
# The easiest option here to make them uniform will be to merge the two channels in the stero samples into one by averaging the values of the two channels. 

# In[ ]:


# num of channels 

print(audiodf.num_channels.value_counts(normalize=True))


# #### Sample rate 
# 
# The rate of capture and playback is called the sample rate. There is a wide range of Sample rates that have been used across all the samples which is a concern (ranging from 96k to 8k).
# 
# This likley means that we will have to apply a sample-rate conversion technique (either up-conversion or down-conversion) so we can see an agnostic representation of their waveform which will allow us to do a fair comparison. 

# In[ ]:


# sample rates 

print(audiodf.sample_rate.value_counts(normalize=True))


# #### Bit-depth 
# 
# Bit depth is the number of bits available for each sample. The higher the bit depth, the higher the quality of the audio. There is also a wide range of bit-depths. It's likely that we may need to normalise them by taking the maximum and minimum amplitude values for a given bit-depth. 

# In[ ]:


# bit depth

print(audiodf.bit_depth.value_counts(normalize=True))


# #### Other audio properties to consider 
# 
# We may also need to consider normalising the volume levels (wave amplitude value) if this is seen to vary greatly, by either looking at the peak volume or the RMS volume. 
# 
# *In the next part we will preprocess the data*

# In[ ]:


#@title Question: What purposes can *librosa* library serve? { run: "auto" }

Options = "Image processing" #@param ["Image processing", "Speech Processing", "Natural language processing"]
print("Correct answer = Speech Processing")


# ## PART 2. Data Preprocessing and Data Splitting

# ### Audio properties that will require normalising 
# 
# Following on from the previous part of notebook, we identifed the following audio properties that need preprocessing to ensure consistency across the whole dataset:  
# 
# - Audio Channels 
# - Sample rate 
# - Bit-depth
# 
# We will continue to use Librosa which will be useful for the pre-processing and feature extraction. 

# ### Preprocessing stage 
# 
# For much of the preprocessing we will be able to use [Librosa's load() function.](https://librosa.github.io/librosa/generated/librosa.core.load.html) 
# 
# We will compare the outputs from Librosa against the default outputs of [scipy's wavfile library](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html) using a chosen file from the dataset. 

# #### Sample rate conversion 
# 
# By default, Librosa’s load function converts the sampling rate to 22.05 KHz which we can use as our (human) comparison level. 

# In[ ]:


import librosa 
from scipy.io import wavfile as wav
import numpy as np

filename = 'UrbanSound Dataset sample/audio/100852-0-0-0.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate) 


# #### Bit-depth 
# 
# Librosa’s load function will also normalise the data so it's values range between -1 and 1. This removes the complication of the dataset having a wide range of bit-depths. 

# In[ ]:


print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))


# #### Merge audio channels 
# 
# Librosa will also convert the signal to mono, meaning the number of channels will always be 1. 

# In[ ]:


import matplotlib.pyplot as plt

# Original audio with 2 channels 
plt.figure(figsize=(12, 4))
plt.plot(scipy_audio)


# In[ ]:


# Librosa audio with channels merged 
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio)


# #### Other audio properties to consider
# 
# At this stage it is not yet clear whether other factors may also need to be taken into account, such as sample duration length and volume levels. 
# 
# We will proceed as is for the meantime and come back to address these later if it's perceived to be effecting the validity of our target metrics. 

# ### Extract Features 
# 
# As outlined, we will extract [Mel-Frequency Cepstral Coefficients (MFCC)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) from the the audio samples. 
# 
# The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. 
# 
# ![alt text](https://cdn-images-1.medium.com/max/1200/1*7sKM9aECRmuoqTadCYVw9A.jpeg)
# 
# These audio representations will allow us to identify features for classification.
# 
# ![alt text](https://i.ibb.co/8mws07n/image.png)  
# 
# TL; DR — MFCC features represent phonemes (distinct units of sound) 

# #### Extracting a MFCC
# 
# For this we will use [Librosa's mfcc() function](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html) which generates an MFCC from time series audio data. 

# In[ ]:


mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)

# y:audio time series
# sr:sampling rate of y
# n_mfcc:number of MFCCs to return

print(mfccs.shape)


# This shows librosa calculated a series of 40 MFCCs over 173 frames. 

# In[ ]:


import librosa.display
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')


# #### Extracting MFCC's for every file 
# 
# We will now extract an MFCC for each audio file in the dataset and store it in a Panda Dataframe along with it's classification label. 

# `extract_features` takes file path as input, read the file by calling librosa.load method, extract and return features

# Below methods are all that is required to convert raw sound clips into informative features (along with a class label for each sound clip) that we can directly feed into our classifier.

# In[ ]:


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # res_type: resample type, resampy’s high-quality mode is ‘kaiser_best’
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled


# In[ ]:


# Load various imports 
import pandas as pd
import os
import librosa
import time
import numpy as np

# Set the path to the full UrbanSound dataset 
fulldatasetpath = 'UrbanSound8K/audio/'

metadata = pd.read_csv('UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

features = []

tic = time.time()
# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files') 
toc = time.time()
print('Time taken to run', toc-tic)


# In[ ]:


# Saving features to drive for later use (Optional)
# featuresdf.to_pickle("features_new.pkl")


# In[ ]:


# Downloading features to local system (Optional)
# from google.colab import files

# files.download('features_new.pkl')


# ### Convert the data and labels
# 
# We will use `sklearn.preprocessing.LabelEncoder` to encode the categorical text data into model-understandable numerical data. 
# 
# [ *Jupyter Trick* ] Remember the shortcut to check the definition of any function in jupyter notebook?
# 
# Type `? LabelEncoder()` in code & run and you see documentation of it. Cool ahh?

# In[ ]:


# Check documentation in notebook for any function
get_ipython().run_line_magic('pinfo', 'LabelEncoder')


# In[ ]:


# Reading features from drive (Optional)
# featuresdf = pd.read_pickle("features_new.pkl")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 


# ### Split the dataset
# 
# Here we will use `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets. The testing set size will be 20% and we will set a random state (for reproducibility). 
# 

# In[ ]:


# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# ### Store the preprocessed data 

# In[ ]:


### store the preprocessed data for use in the next notebook

get_ipython().run_line_magic('store', 'x_train')
get_ipython().run_line_magic('store', 'x_test')
get_ipython().run_line_magic('store', 'y_train')
get_ipython().run_line_magic('store', 'y_test')
get_ipython().run_line_magic('store', 'yy')
get_ipython().run_line_magic('store', 'le')


# *In the next part we will develop our model*

# In[ ]:


#@title Question: Why do we extract MFCC from audio samples?  { run: "auto" }

Options = "To analyze the frequency characteristics of the sound" #@param ["To summarize the frequency distribution across the window size", "To analyze the frequency characteristics of the sound", "To analyze the time characteristics of the sound","All of the above"]
print("Correct answer = All of the above")


# ## PART 3. Model Training and Evaluation 

# ### Load Preprocessed data 

# In[ ]:


# retrieve the preprocessed data from previous part of notebook

get_ipython().run_line_magic('store', '-r x_train')
get_ipython().run_line_magic('store', '-r x_test')
get_ipython().run_line_magic('store', '-r y_train')
get_ipython().run_line_magic('store', '-r y_test')
get_ipython().run_line_magic('store', '-r yy')
get_ipython().run_line_magic('store', '-r le')


# *Real Time Visualization Tools* -   `Weights & Biases`

# In[ ]:


# Use Weights & Biases to track all runs (Includes all parameters of models - Accuracy, Loss and what not!)
get_ipython().system('pip install --upgrade wandb')


# In[ ]:


# Login
get_ipython().system('wandb login')


# In[ ]:


# logging code
import wandb
from wandb.keras import WandbCallback
wandb.init(project="audio_cnn")


# *Open this link*

# ### Initial model architecture - MLP
# 
# We will start with constructing a Multilayer Perceptron (MLP) Neural Network using Keras and a Tensorflow backend. 
# 
# Starting with a `sequential` model so we can build the model layer by layer. 
# 
# We will begin with a simple model architecture, consisting of five layers, an input layer, 3 hidden layers and an output layer. All layers will be of the `dense` layer type which is a standard layer type that is used in many cases for neural networks. 
# 
# The first layer will receive the input shape. As each sample contains 40 MFCCs (or columns) we have a shape of (1x40) this means we will start with an input shape of 40. 
# 
# The hidden layers will have 256 nodes. The activation function we will be using for our hidden layers is the `ReLU`, or `Rectified Linear Activation`. This activation function has been proven to work well in neural networks.
# 
# We will also apply a `Dropout` value of 50% on our first two layers. This will randomly exclude nodes from each update cycle which in turn results in a network that is capable of better generalisation and is less likely to overfit the training data.
# 
# Our output layer will have 10 nodes (num_labels) which matches the number of possible classifications. The activation is for our output layer is `softmax`. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.
# 
# ![alt text](https://i.ibb.co/gJsprsW/audio-cnn.png)

# In[ ]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_labels = yy.shape[1]

# Construct model 
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))


# ### Compiling the model 
# 
# For compiling our model, we will use the following three parameters: 
# 
# * Loss function - we will use `categorical_crossentropy`. This is the most common choice for classification. A lower score indicates that the model is performing better.
# 
# * Metrics - we will use the `accuracy` metric which will allow us to view the accuracy score on the validation data when we train the model. 
# 
# * Optimizer - here we will use `adam` which is a generally good optimizer for many use cases.
# 

# In[ ]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# In[ ]:


# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# ### Training 
# 
# Here we will train the model. 
# 
# We will start with 100 epochs which is the number of times the model will cycle through the data. The model will improve on each cycle until it reaches a certain point. 
# 
# We will also start with a low batch size, as having a large batch size can reduce the generalisation ability of the model. 

# In[ ]:


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

# Save the model after every epoch
checkpointer = ModelCheckpoint(filepath='Notebooks/saved_models/weights.best.basic_mlp.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()
# Additional callback - WandbCallback() for real-time visualization
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer,WandbCallback()], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# *Open the Wandb Link and see the real-time metrics of the model just like below*

# ![alt text](https://i.imgur.com/gFUxFCj.png)

# ### Test the model 
# 
# Here we will review the accuracy of the model on both the training and test data sets. 

# In[ ]:


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# The initial Training and Testing accuracy scores are quite high. As there is not a great difference between the Training and Test scores (~5%) this suggests that the model has not suffered from large overfitting. 

# ### Predictions  
# 
# Here we will build a method which will allow us to test the models predictions on a specified audio .wav file. 

# In[ ]:


# Same Function again
import librosa 
import numpy as np 

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])


# In[ ]:


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

    predicted_vector = model.predict_classes(prediction_feature)
    # Transform labels back to original encoding
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    # predict probabilities of each class
    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


# ### Validation 
# 
# #### Test with sample data 
# 
# Initial sanity check to verify the predictions using a subsection of the sample audio files we explored in the first notebook. We expect the bulk of these to be classified correctly. 

# In[ ]:


# Class: Air Conditioner

filename = 'UrbanSound Dataset sample/audio/100852-0-0-0.wav' 
print_prediction(filename)


# In[ ]:


# Class: Drilling

filename = 'UrbanSound Dataset sample/audio/103199-4-0-0.wav'
print_prediction(filename) 


# In[ ]:


# Class: Street music 

filename = 'UrbanSound Dataset sample/audio/101848-9-0-0.wav'
print_prediction(filename) 


# In[ ]:


# Class: Car Horn 

filename = 'UrbanSound Dataset sample/audio/100648-1-0-0.wav'
print_prediction(filename) 


# #### Observations 
# 
# From this brief sanity check the model seems to predict well. One errror was observed whereby a car horn was incorrectly classifed as a street music. 
# 
# We can see from the per class confidence that this was quite a low score (43%). This allows follows our early observation that a street music and car horn are similar in spectral shape. 

# ### Other audio
# 
# Here we will use a sample of various copyright free sounds that were not part of either our test or training data to further validate our model. 

# In[ ]:


filename = 'Evaluation audio/dog_bark_1.wav'
print_prediction(filename)


# In[ ]:


filename = 'Evaluation audio/drilling_1.wav'

print_prediction(filename) 


# In[ ]:


filename = 'Evaluation audio/gun_shot_1.wav'

print_prediction(filename)

# sample data weighted towards gun shot - peak in the dog barking sample is simmilar in shape to the gun shot sample


# In[ ]:


filename = 'Evaluation audio/siren_1.wav'

print_prediction(filename)


# #### Observations 
# 
# The performance of our initial model is satisfactorry and has generalised well, seeming to predict well when tested against new audio data. 
# 
# *In the next part we will refine our model*

# In[ ]:


#@title Question: Which loss function is appropriate for multi-class problem? { run: "auto" }

Options = "Binary Cross-entropy" #@param ["Binary Cross-entropy", "Mean Squared Error", "Categorical Cross-entropy"]
print("Correct answer = Categorical Cross-entropy")


# ## PART 4. Model Refinement 

# ### Load Preprocessed data 

# In[ ]:


# retrieve the preprocessed data from previous notebook

get_ipython().run_line_magic('store', '-r x_train')
get_ipython().run_line_magic('store', '-r x_test')
get_ipython().run_line_magic('store', '-r y_train')
get_ipython().run_line_magic('store', '-r y_test')
get_ipython().run_line_magic('store', '-r yy')
get_ipython().run_line_magic('store', '-r le')


# ### Model refinement
# 
# In our inital attempt, we were able to achieve a Classification Accuracy score of: 
# 
# * Training data Accuracy:  ~90% 
# * Testing data Accuracy:  ~85% 
# 
# We will now see if we can improve upon that score using a Convolutional Neural Network (CNN). Also this tells the `importance of CNNs` over other shallow networks like MLPs

# Remember CNNs
# 
# ![alt text](https://cdn-images-1.medium.com/max/800/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)

# ![alt text](https://i.ibb.co/X7Sv7y2/image.png)

# Note: If you want to get in-depth understanding of how CNN works. Kindly consult these resources:  [[1]](http://cs231n.github.io/convolutional-networks),  [[2]](http://arxiv.org/pdf/1603.07285v1.pdf) and  [[3]](http://www.deeplearningbook.org/contents/convnets.html).

# ### Feature Extraction refinement 
# 
# In the prevous feature extraction stage, the MFCC vectors would vary in size for the different audio files (depending on the samples duration). 
# 
# However, CNNs require a fixed size for all inputs. To overcome this we will zero pad the output vectors to make them all the same size. 

# In[ ]:


# Notice small modification (Padding) made to the same function
import numpy as np
max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


# In[ ]:


# Same Function Again
# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = 'UrbanSound8K/audio/'

metadata = pd.read_csv('UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files') 


# In[ ]:


# Saving features to drive for later use
# featuresdf.to_pickle("features_refined.pkl")


# In[ ]:


# Downloading features to local system
# from google.colab import files

# files.download('features_refined.pkl')


# In[ ]:


# Reading features from drive
# import pandas as pd
# featuresdf = pd.read_pickle("features_refined.pkl")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# ### Convolutional Neural Network (CNN) model architecture 
# 
# 
# We will modify our model to be a Convolutional Neural Network (CNN) again using Keras and a Tensorflow backend. 
# 
# Again we will use a `sequential` model, starting with a simple model architecture, consisting of four `Conv2D` convolution layers, with our final output layer being a `dense` layer. 
# 
# The convolution layers are designed for feature detection. It works by sliding a filter window over the input and performing a matrix multiplication and storing the result in a feature map. This operation is known as a convolution. 
# 
# 
# The `filter` parameter specifies the number of nodes in each layer. Each layer will increase in size from 16, 32, 64 to 128, while the `kernel_size` parameter specifies the size of the kernel window which in this case is 2 resulting in a 2x2 filter matrix. 
# 
# The first layer will receive the input shape of (40, 174, 1) where 40 is the number of MFCC's 174 is the number of frames taking padding into account and the 1 signifying that the audio is mono. 
# 
# The activation function we will be using for our convolutional layers is `ReLU` which is the same as our previous model. We will use a smaller `Dropout` value of 20% on our convolutional layers. 
# 
# Each convolutional layer has an associated pooling layer of `MaxPooling2D` type with the final convolutional layer having a `GlobalAveragePooling2D` type. The pooling layer is do reduce the dimensionality of the model (by reducing the parameters and subsquent computation requirements) which serves to shorten the training time and reduce overfitting. The Max Pooling type takes the maximum size for each window and the Global Average Pooling type takes the average which is suitable for feeding into our `dense` output layer.  
# 
# Our output layer will have 10 nodes (num_labels) which matches the number of possible classifications. The activation is for our output layer is `softmax`. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.

# In[ ]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

# Shape produced by MFCC on raw sound
num_rows = 40
num_columns = 174
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax')) 


# ### Compiling the model 
# 
# For compiling our model, we will use the same three parameters as the previous model: 

# In[ ]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# In[ ]:


# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# ### Training 
# 
# Here we will train the model. As training a CNN can take a sigificant amount of time, we will start with a low number of epochs and a low batch size. If we can see from the output that the model is converging, we will increase both numbers.  

# In[ ]:


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

#num_epochs = 12
#num_batch_size = 128

num_epochs = 72
num_batch_size = 256

# Save the model after every epoch
checkpointer = ModelCheckpoint(filepath='Notebooks/saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer,WandbCallback()], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# ### Test the model 
# 
# Here we will review the accuracy of the model on both the training and test data sets. 

# In[ ]:


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# The Training and Testing accuracy scores are both high and an increase on our initial model. Training accuracy has increased by ~6% and Testing accuracy has increased by ~4%. 
# 
# There is a marginal increase in the difference between the Training and Test scores (~6% compared to ~5% previously) though the difference remains low so the model has not suffered from large overfitting. 

# ### Predictions  
# 
# Here we will modify our previous method for testing the models predictions on a specified audio .wav file. 

# In[ ]:


# Same function with small modification
def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    # reshaping to predict for one input
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


# ### Validation 
# 
# #### Test with sample data 
# 
# As before we will verify the predictions using a subsection of the sample audio files we explored in the first notebook. We expect the bulk of these to be classified correctly. 

# In[ ]:


# Class: Air Conditioner

filename = 'UrbanSound Dataset sample/audio/100852-0-0-0.wav' 
print_prediction(filename) 


# In[ ]:


# Class: Drilling

filename = 'UrbanSound Dataset sample/audio/103199-4-0-0.wav'
print_prediction(filename) 


# In[ ]:


# Class: Street music 

filename = 'UrbanSound Dataset sample/audio/101848-9-0-0.wav'
print_prediction(filename) 


# In[ ]:


# Class: Car Horn 

filename = 'UrbanSound Dataset sample/audio/100648-1-0-0.wav'
print_prediction(filename) 


# #### Observations 
# 
# We can see that the model performs well. 
# 
# Interestingly, car horn was again incorrectly classifed but this time as dog bark - though the per class confidence shows it was a close decision between car horn with ~20% confidence and dog bark at ~30% confidence.  

# ### Other audio
# 
# Again we will further validate our model using a sample of various copyright free sounds that we not part of either our test or training data. 

# In[ ]:


filename = 'Evaluation audio/dog_bark_1.wav'
print_prediction(filename) 


# In[ ]:


filename = 'Evaluation audio/drilling_1.wav'

print_prediction(filename) 


# In[ ]:


filename = 'Evaluation audio/gun_shot_1.wav'

print_prediction(filename) 


# **Confusion Matrix**
# 
# What observations can you draw from the matrix below?
# 
# *Try getting it for yourself too!*
# 
# ![alt text](https://cdn-images-1.medium.com/max/800/1*wVC7H1sQw1Id872xuk9VDg.png)

# #### Observations 
# 
# The performance of our final model is very good and has generalised well, seeming to predict well when tested against new audio data. 

# In[ ]:


#@title Question: Which Neural network  architecture outperforms on urbansound8K dataset?? { run: "auto" }

Options = "SVM" #@param ["MLP", "CNN", "SVM"]
print("Correct answer = CNN")


# # Enjoy this Fun "[Neural Beatbox](https://codepen.io/setiadeepanshu/full/PrQmNP)" based on audio classification using CNN
# 
# 
# 
# > **Wanna DEPLOY your models live on web. Stay tuned for next session!!**
# 
# > **Wanna Learn how each layer of our model SEES the audio!!**
# 
# 

# # Future steps to explore
# 
# Now that we saw a simple applications, we can ideate a few more methods which can help us improve our score
# 
# 1.   We applied a simple neural network model to the problem. Our immediate next step should be to understand where does the model fail and why. By this, we want to conceptualize our understanding of the failures of algorithm so that the next time we build a model, it does not do the same mistakes
# 2.   We can build more efficient models than our “better models”, such as recurrent neural networks. These models have been proven to solve such problems with greater ease.
# 3.   We touched the concept of data augmentation, but we did not apply them here. You could try it to see if it works for the problem. It generally works!
# 
# 
# 
# 
# 

# # Conclusion
# 
# The process used for this project can be summarised with the following steps:
# 
# 1. The initial problem was defined and relevant public dataset was located.
# 2. The data was explored and analysed.
# 3. Data was preprocessed and features were extracted.
# 4. An initial model was trained and evaluated.
# 5. A further model was trained and refined.
# 6. The final model was evaluated.

# # Assignment 
# 
# 
# 
# ## 1.   `Speaker Identification Using CNN ` - The code is present  [here](https://github.com/SETIADEEPANSHU/Audio_Classification_Using_CNN/blob/master/Notebooks/Video_Example_conan_or_colbert.ipynb) . You have to run this notebook with your dataset of two speakers and identify them. Submit your github repository links. More Hints in the code! (500 Points)
# 

# # References
# 
# [1] Justin Salamon, Christopher Jacoby and Juan Pablo Bello. Urban Sound Datasets. https://urbansounddataset.weebly.com/urbansound8k.html
# 
# [2] Justin Salamon and Juan Pablo Bello. Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. https://arxiv.org/pdf/1608.04363.pdf
# 
# [3] Mel-frequency cepstrum Wikipedia page https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
# 
# [4] J. Salamon, C. Jacoby, and J. P. Bello. A dataset and taxonomy for urban sound research. http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf
# 
# [5] Manik Soni AI which classifies Sounds:Code:Python. https://hackernoon.com/ai-which-classifies-sounds-code-python-6a07a2043810
# 
# [6] Manash Kumar Mandal Building a Dead Simple Speech Recognition
# Engine using ConvNet in Keras. https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
# 
# [7] Eijaz Allibhai Building a Convolutional Neural Network (CNN) in Keras. https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# 
# [8] Daphne Cornelisse An intuitive guide to Convolutional Neural Networks. https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050
# 
# [9] Urban Sound Classification - Part 2: sample rate conversion, Librosa. https://towardsdatascience.com/urban-sound-classification-part-2-sample-rate-conversion-librosa-ba7bc88f209a
# 
# [10] Wavsource.com THE Source for Free Sound Files and Reviews. http://www.wavsource.com
# 
# [11] Soundbible.com. http://soundbible.com
