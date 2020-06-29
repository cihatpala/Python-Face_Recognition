from imutils import paths
import face_recognition
import pickle
import cv2
import os


print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("dataset"))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	while True:
		try:
			boxes = face_recognition.face_locations(rgb, model="cnn")  # or hog
			break
		except:
			rgb = cv2.resize(rgb, (int(rgb.shape[0]/2), int(rgb.shape[1]/2)))

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings_multi.pickle", "wb")
f.write(pickle.dumps(data))
f.close()