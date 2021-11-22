#Parte 2: Crear algoritmo de reconocimiento facial, que detecte y  reconozca un rostro conocido.

#importamos libreria opencv para trabajar con reconocimiento facial
import cv2
#importamos libreria numpy para tener un mayor manejo de vectores y matrices
import numpy as np

#en esta funcion hacemos uso de la camara del dispositivo
cap = cv2.VideoCapture(0)

#aqui hacemos uso de machine learning usando el clasificador de rostros haarcascade
# pre-entrenado proporcionado por opencv
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#aqui aplicamos el clasificador y le pasamos los parametros para que pueda detectar rostros 
	faces = faceClassif.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow('frame',frame)
	
  #presionando la letra q se detiene la aplicacion.  
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()