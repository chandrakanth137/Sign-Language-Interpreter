# SAY-GUY

Say-Guy is a real time sign language interpreter that is used to bridge the gap between the vocally muted and the rest of the world.

It is a deep learning based model embedded in a web application which takes live web camera feed as input and outputs the performed sign language as text on the screen.

Say-Guy uses a CNN LSTM based deep learning model architecture to detect the action performed in each frame and after the action is completely performed, the value associated with it is displayed.

Flask is used as the framework to run it as web application, the web app is hosted using ng-rock and the model was developed using Tensorflow, Keras, Mediapipe and OpenCV. 

## Installation

#### To install the required libraries
```
pip install -r requirements.txt
```

#### To run the app
```
flask run
```

in the cloned / downloaded directory

link to the guthub repository
https://github.com/chandrakanth137/Sign-Language-Interpreter


## Working

__app.py__ is the main driver file that initiates the process of the web application.

__helper.py__ is used to extract keypoints from the live camera feed and return it to model for further processing.

__actions.csv__ contains the basic terminologies to classify the actions.

**conv-rnn-model.h5** contains the saved keras model which is a Convolutional-Recurrent(LSTM) hybrid model

![model arch.jpeg]

The camera feed captures the video stream which is then passed to the mediapipe holistic network

The holistic network outputs an processed image consisting simultaneous pose, face and landmark estimations. 

The keypoints from the processed results are extracted, which are then passed to the conv-rnn model to predict the meaning of the given sign.






