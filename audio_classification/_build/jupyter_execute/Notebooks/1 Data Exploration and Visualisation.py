#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration and Visualisation

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

# ### Auditory inspection 
# 
# We will use `IPython.display.Audio` to play the audio files so we can inspect aurally. 

# In[1]:


import IPython.display as ipd

ipd.Audio('../UrbanSound Dataset sample/audio/100032-3-0-0.wav')


# ### Visual inspection
# 
# We will load a sample from each class and visually inspect the data for any patterns. We will use librosa to load the audio file into an array then librosa.display and matplotlib to display the waveform. 

# In[2]:


# Load imports

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# In[3]:


# Class: Air Conditioner

filename = '../UrbanSound Dataset sample/audio/100852-0-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[4]:


# Class: Car horn 

filename = '../UrbanSound Dataset sample/audio/100648-1-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[5]:


# Class: Children playing 

filename = '../UrbanSound Dataset sample/audio/100263-2-0-117.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[6]:


# Class: Dog bark

filename = '../UrbanSound Dataset sample/audio/100032-3-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[7]:


# Class: Drilling

filename = '../UrbanSound Dataset sample/audio/103199-4-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[8]:


# Class: Engine Idling 

filename = '../UrbanSound Dataset sample/audio/102857-5-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[9]:


# Class: Gunshot

filename = '../UrbanSound Dataset sample/audio/102305-6-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[10]:


# Class: Jackhammer

filename = '../UrbanSound Dataset sample/audio/103074-7-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[11]:


# Class: Siren

filename = '../UrbanSound Dataset sample/audio/102853-8-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[12]:


# Class: Street music

filename = '../UrbanSound Dataset sample/audio/101848-9-0-0.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# ### Observations 
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

# ### Dataset Metadata 
# 
# Here we will load the UrbanSound metadata .csv file into a Panda dataframe. 

# In[13]:


import pandas as pd
metadata = pd.read_csv('../UrbanSound Dataset sample/metadata/UrbanSound8K.csv')
metadata.head()


# ### Class distributions

# In[14]:


print(metadata.class_name.value_counts())


# ### Observations 
# 
# Here we can see the Class labels are unbalanced. Although 7 out of the 10 classes all have exactly 1000 samples, and siren is not far off with 929, the remaining two (car_horn, gun_shot) have significantly less samples at 43% and 37% respectively. 
# 
# This will be a concern and something we may need to address later on. 

# ### Audio sample file properties
# 
# Next we will iterate through each of the audio sample files and extract, number of audio channels, sample rate and bit-depth. 

# In[15]:


# Load various imports 
import pandas as pd
import os
import librosa
import librosa.display

from helpers.wavfilehelper import WavFileHelper
wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath('/Volumes/Untitled/ML_Data/Urban Sound/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])


# ### Audio channels 
# 
# Most of the samples have two audio channels (meaning stereo) with a few with just the one channel (mono). 
# 
# The easiest option here to make them uniform will be to merge the two channels in the stero samples into one by averaging the values of the two channels. 

# In[19]:


# num of channels 

print(audiodf.num_channels.value_counts(normalize=True))


# ### Sample rate 
# 
# There is a wide range of Sample rates that have been used across all the samples which is a concern (ranging from 96k to 8k).
# 
# This likley means that we will have to apply a sample-rate conversion technique (either up-conversion or down-conversion) so we can see an agnostic representation of their waveform which will allow us to do a fair comparison. 

# In[21]:


# sample rates 

print(audiodf.sample_rate.value_counts(normalize=True))


# ### Bit-depth 
# 
# There is also a wide range of bit-depths. It's likely that we may need to normalise them by taking the maximum and minimum amplitude values for a given bit-depth. 

# In[22]:


# bit depth

print(audiodf.bit_depth.value_counts(normalize=True))


# ### Other audio properties to consider 
# 
# We may also need to consider normalising the volume levels (wave amplitude value) if this is seen to vary greatly, by either looking at the peak volume or the RMS volume. 

# ### *In the next notebook we will preprocess the data*
