#!/usr/bin/env python
# coding: utf-8

# ## Model Training and Evaluation 

# ### Load Preprocessed data 

# In[1]:


# retrieve the preprocessed data from previous notebook

get_ipython().run_line_magic('store', '-r x_train')
get_ipython().run_line_magic('store', '-r x_test')
get_ipython().run_line_magic('store', '-r y_train')
get_ipython().run_line_magic('store', '-r y_test')
get_ipython().run_line_magic('store', '-r yy')
get_ipython().run_line_magic('store', '-r le')


# ### Initial model architecture - MLP
# 
# We will start with constructing a Multilayer Perceptron (MLP) Neural Network using Keras and a Tensorflow backend. 
# 
# Starting with a `sequential` model so we can build the model layer by layer. 
# 
# We will begin with a simple model architecture, consisting of three layers, an input layer, a hidden layer and an output layer. All three layers will be of the `dense` layer type which is a standard layer type that is used in many cases for neural networks. 
# 
# The first layer will receive the input shape. As each sample contains 40 MFCCs (or columns) we have a shape of (1x40) this means we will start with an input shape of 40. 
# 
# The first two layers will have 256 nodes. The activation function we will be using for our first 2 layers is the `ReLU`, or `Rectified Linear Activation`. This activation function has been proven to work well in neural networks.
# 
# We will also apply a `Dropout` value of 50% on our first two layers. This will randomly exclude nodes from each update cycle which in turn results in a network that is capable of better generalisation and is less likely to overfit the training data.
# 
# Our output layer will have 10 nodes (num_labels) which matches the number of possible classifications. The activation is for our output layer is `softmax`. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.

# In[2]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
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

# In[3]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# In[4]:


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

# In[5]:


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_mlp.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# ### Test the model 
# 
# Here we will review the accuracy of the model on both the training and test data sets. 

# In[6]:


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# The initial Training and Testing accuracy scores are quite high. As there is not a great difference between the Training and Test scores (~5%) this suggests that the model has not suffered from overfitting. 

# ### Predictions  
# 
# Here we will build a method which will allow us to test the models predictions on a specified audio .wav file. 

# In[7]:


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


# In[8]:


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

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
# Initial sainity check to verify the predictions using a subsection of the sample audio files we explored in the first notebook. We expect the bulk of these to be classified correctly. 

# In[9]:


# Class: Air Conditioner

filename = '../UrbanSound Dataset sample/audio/100852-0-0-0.wav' 
print_prediction(filename) 


# In[10]:


# Class: Drilling

filename = '../UrbanSound Dataset sample/audio/103199-4-0-0.wav'
print_prediction(filename) 


# In[11]:


# Class: Street music 

filename = '../UrbanSound Dataset sample/audio/101848-9-0-0.wav'
print_prediction(filename) 


# In[12]:


# Class: Car Horn 

filename = '../UrbanSound Dataset sample/audio/100648-1-0-0.wav'
print_prediction(filename) 


# #### Observations 
# 
# From this brief sanity check the model seems to predict well. One errror was observed whereby a car horn was incorrectly classifed as a dog bark. 
# 
# We can see from the per class confidence that this was quite a low score (43%). This allows follows our early observation that a dog bark and car horn are similar in spectral shape. 

# ### Other audio
# 
# Here we will use a sample of various copyright free sounds that we not part of either our test or training data to further validate our model. 

# In[13]:


filename = '../Evaluation audio/dog_bark_1.wav'
print_prediction(filename) 


# In[14]:


filename = '../Evaluation audio/drilling_1.wav'

print_prediction(filename) 


# In[15]:


filename = '../Evaluation audio/gun_shot_1.wav'

print_prediction(filename) 

# sample data weighted towards gun shot - peak in the dog barking sample is simmilar in shape to the gun shot sample


# In[16]:


filename = '../Evaluation audio/siren_1.wav'

print_prediction(filename) 


# #### Observations 
# 
# The performance of our initial model is satisfactorry and has generalised well, seeming to predict well when tested against new audio data. 

# ### *In the next notebook we will refine our model*
