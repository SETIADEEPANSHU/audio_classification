#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing and Data Splitting

# ### Audio properties that will require normalising 
# 
# Following on from the previous notebook, we identifed the following audio properties that need preprocessing to ensure consistency across the whole dataset:  
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
# 
# #### Sample rate conversion 
# 
# By default, Librosa’s load function converts the sampling rate to 22.05 KHz which we can use as our comparison level. 

# In[2]:


import librosa 
from scipy.io import wavfile as wav
import numpy as np

filename = '../UrbanSound Dataset sample/audio/100852-0-0-0.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate) 


# #### Bit-depth 
# 
# Librosa’s load function will also normalise the data so it's values range between -1 and 1. This removes the complication of the dataset having a wide range of bit-depths. 

# In[3]:


print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))


# #### Merge audio channels 
# 
# Librosa will also convert the signal to mono, meaning the number of channels will always be 1. 

# In[5]:


import matplotlib.pyplot as plt

# Original audio with 2 channels 
plt.figure(figsize=(12, 4))
plt.plot(scipy_audio)


# In[6]:


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
# As outlined in the proposal, we will extract [Mel-Frequency Cepstral Coefficients (MFCC)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) from the the audio samples. 
# 
# The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. These audio representations will allow us to identify features for classification. 

# #### Extracting a MFCC
# 
# For this we will use [Librosa's mfcc() function](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html) which generates an MFCC from time series audio data. 

# In[7]:


mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)


# This shows librosa calculated a series of 40 MFCCs over 173 frames. 

# In[8]:


import librosa.display
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')


# #### Extracting MFCC's for every file 
# 
# We will now extract an MFCC for each audio file in the dataset and store it in a Panda Dataframe along with it's classification label. 

# In[14]:


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled


# In[19]:


# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = '/Volumes/Untitled/ML_Data/Urban Sound/UrbanSound8K/audio/'

metadata = pd.read_csv('../UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

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


# ### Convert the data and labels
# 
# We will use `sklearn.preprocessing.LabelEncoder` to encode the categorical text data into model-understandable numerical data. 

# In[16]:


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
# Here we will use `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets. The testing set size will be 20% and we will set a random state. 
# 

# In[17]:


# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# ### Store the preprocessed data 

# In[18]:


### store the preprocessed data for use in the next notebook

get_ipython().run_line_magic('store', 'x_train')
get_ipython().run_line_magic('store', 'x_test')
get_ipython().run_line_magic('store', 'y_train')
get_ipython().run_line_magic('store', 'y_test')
get_ipython().run_line_magic('store', 'yy')
get_ipython().run_line_magic('store', 'le')


# ### *In the next notebook we will develop our model*
