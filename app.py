from flask import Flask, render_template, Response, session
import cv2
import mediapipe as mp
import tensorflow as tf
import csv
from helper import *

app = Flask(__name__)

cap = cv2.VideoCapture(0) 

def gen_frames():  # generate frame by frame from camera
    sequence = []
    sequence_length = 40
    threshold = 0.90
    f = open('actions.csv', 'r')
    reader = csv.reader(f)
    actions = []
    for word in reader:
        actions.append(word[0])
    f.close()
    sentence = []
    model = tf.keras.models.load_model("conv-rnn-model.h5")
    while True:
        # Capture frame-by-frame
        with mp_holistic.Holistic(min_detection_confidence=0.77, min_tracking_confidence=0.77) as holistic:
            success, frame = cap.read()
            if not success:
                break
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold: 
                    tmp = actions[np.argmax(res)]
                    if(tmp == 'Idle'):
                        sentence = [tmp]
                    if(len(sentence) > 0):
                        if(not(sentence[-1] == tmp)):
                            sentence.append(tmp)
                    else:
                        sentence.append(tmp)
                    if(len(sentence) > 1 and sentence[-2] == 'Idle' and tmp != 'Idle'):
                            del sentence[-2]             
                if len(sentence) > 3: 
                    sentence = sentence[-1:]
                    
            cv2.putText(image,' '.join(sentence), (3,85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            ret,buffer=cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
