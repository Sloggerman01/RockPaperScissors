# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = cv2.resize(frame, (224,224))
    test_image = np.array(img,dtype=np.float32)
    test_img = np.expand_dims(test_image,axis=0)
    normalized_image = test_image/255.0
    prediction = model.predict(normalized_image)
    print("Prediction: ", prediction)
    # Display the resulting frame
    rock = int(prediction[0][0]*100)
    paper = int(prediction[0][1]*100)
    scissors = int(prediction[0][2]*100)
    print(f"rock: {rock} %,paper: {paper}%,scissors:{scissors}%")
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()