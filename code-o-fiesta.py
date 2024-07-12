# import cv2  
# import pyttsx3

# text_speech= pyttsx3.init() 


# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)

# classNames= []
# classFile = "coco.names"
# with open(classFile,'rt') as f: 
#     classNames =f.read().rstrip('\n').split('\n')

# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'frozen_inference_graph.pb'

# net = cv2.dnn_DetectionModel(weightsPath,configPath)

# net.setInputSize (320,320)
# net.setInputScale(1.0/ 127.5) 
# net. setInputMean ((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)

# while True:
#      success,img=cap.read()
#      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
#      print(classIds, bbox)
     
#      if len(classIds) != 0:
#         for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
#             cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#             cv2.putText(img, classNames[classId-1].upper(), (box [0]+10,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255.0),2) 
#      cv2.imshow("output",img)
#      cv2.waitKey(1)
#      answer= classNames[classId-1]
#      newVoiceRate =40
#      text_speech.setProperty('rate',newVoiceRate)
#      text_speech.say(answer)
#      text_speech.runAndWait()

import cv2
import pyttsx3
import matplotlib.pyplot as plt

text_speech = pyttsx3.init() 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f: 
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # Convert the image from BGR to RGB format for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Output")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

    if len(classIds) != 0:
        answer = classNames[classIds[0]-1]
        newVoiceRate = 40
        text_speech.setProperty('rate', newVoiceRate)
        text_speech.say(answer)
        text_speech.runAndWait()

    # Add a condition to break the loop (e.g., press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






