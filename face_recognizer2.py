import pickle as pk
import cv2
from pickle import dump
import os
import dlib
import face_recognition
import numpy as np
import time

path = os.path.dirname(os.path.abspath(__file__))
def who_is_it(encodings, database, threshold = 0.25):
		"""
		Implements face recognition for the happy house by finding who is the person on the image_path image.

		Arguments:
		image_path -- path to an image
		database -- database containing image encodings along with the name of the person on the image
		model -- your Inception model instance in Keras

		Returns:
		min_dist -- the minimum distance between image_path encoding and the encodings from the database
		identity -- string, the name prediction for the person on image_path
		"""
		results = []
		scores = []

		# Initialize "min_dist" to a large value, say 100 (≈1 line)
		min_dist = 10
		identity = None

	   # create array for db and names
		db_vectors=[]
		db_names=[]
		for (name, db_enc) in database:
			db_vectors.append(db_enc)
			db_names.append(name)

		db_vectors=np.array(db_vectors)
		for encoding in encodings:
    		# Compute L2 distance between the target "encoding" and db_vectors
			dist_vector = np.linalg.norm(db_vectors - np.array(encoding),axis=1)
			# calculating minimum distance and it's corresponding index
			min_dist = np.min(np.array(dist_vector))
			index = np.argmin(np.array(dist_vector))
			identity=db_names[index]

			if min_dist > threshold:

				identity="Unknown"

			results.append(identity)
			scores.append(min_dist)

		return results, scores

class FaceRecognizer:
	""" Class for recognising faces, adding new faces and delete existing faces from the database.
	"""

	def __init__(self):
    	# database path
		self.db_path = os.path.join(path, 'models','db_enc.pkl')
		# load the face detector
		self.detector = dlib.get_frontal_face_detector()

		# loading database
		# if(os.path.exists(self.db_path)):
		# 	self.database = pk.load(open(self.db_path, "rb"))
		# 	print(self.database)
		# else:
		self.database = []
		self.database.append(('name', np.zeros(128,dtype=np.float64)))

	def add_new_face(self, img, name,boxes):
		""" Adds a new face to the database.
		"""
		# convert  BGR to RBG
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# face detection
		#boxes = face_recognition.face_locations(img, model='cnn')
		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(img, boxes)

		for encoding in encodings:
			print(np.array(encoding).shape)
			self.database.append((name, encoding))
			print("Added {} to the database".format(name))

		# Saving database
		#dump(self.database, open(self.db_path, "wb"))

	def delete_a_face(self, name):
		""" Deletes an entry from the database.
		"""
		for i,(db_name, db_enc) in enumerate(self.database):
			print(name,db_name)
			if db_name == name:
				self.database.pop(i)
				print("Removed {} from database".format(name))

		# Saving database
		dump(self.database, open(self.db_path, "wb"))

	def get_faces(self, img):
		box=[]
		names=[]
		#print("[INFO] recognizing faces...")
		# convert  BGR to RBG
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# detecting faces
		boxes = face_recognition.face_locations(img, model='hog')

		start = time.time()
		# get the encodings of detected faces
		encodings = face_recognition.face_encodings(img, boxes)
		end = time.time()
		#print(end - start)
		# Predict output
		names, scores = who_is_it(encodings, self.database, threshold=0.5)

		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):

			box.append([left,top,right,bottom])
			names.append(name)
			# draw the predicted face name on the image
			cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)

		return img,box,names

if ( __name__ == "__main__"):
	images =os.listdir('Webcam')
	for image in images:
		fce=FaceRecognizer()
		camera = cv2.imread('Webcam/'+image)
		fce.add_new_face(camera,'name')
