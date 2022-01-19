from collections import deque
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

def load(M_name):
    Model = tf.keras.models.load_model(M_name)
    return Model


st.title("Violence Detection")

Model = load("Model.h5")
Q = deque(maxlen = 128)
FRAME_WINDOW = st.image([])
W,H = (None,None)

col1,col2 = st.columns(2)


user_input = st.text_input("Enter The Path for Video And Press Start : ")
if col1.button("start"):
    cap = cv2.VideoCapture(user_input)
    while True:
        if  cap.get(cv2.CAP_PROP_POS_FRAMES) ==  cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        success,img = cap.read()
        if not success:
            break

        if W is None or H is None:
            (H,W) = img.shape[:2]

        output = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128)).astype("float32")
        img = img.reshape(128, 128, 3) / 255

        preds = Model.predict(np.expand_dims(img, axis=0))[0]
        Q.append(preds)

        # perform prediction averaging over the current history of
        # previous predictions
        results = np.array(Q).mean(axis=0)
        i = (max(Q) > 0.50)[0]
        # i = (preds > 0.50)[0]
        label = i

        color = (0,255,0)

        if label:
            color = (0,0,255)
        else:
            color = (0,255,0)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX 

        cv2.putText(output, text, (35, 60), FONT,1.6, color, 8) 


        # show the output image
        output = cv2.cvtColor(output , cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(output, use_column_width=True)
if col2.button("stop"):
    fr = st.empty()


