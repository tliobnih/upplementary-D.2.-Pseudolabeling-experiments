import numpy as np

class Channel1to3_v0():     	
	def __call__(self, image):
		image = np.array(image)
		image = image.reshape((1,28,28))
		image = np.vstack((image, image, image))
		image = np.transpose(image, (1,2,0))
		return image
class change_channel():     	
	def __call__(self, image):
		image = np.array(image)
		return np.transpose(image, (1,2,0))
class add_channel_1():     	
	def __call__(self, image):
		image = np.array(image)
		image = [image,1]
		return image
class add_channel_0():     	
	def __call__(self, image):
		image = np.array(image)
		image = [image,0]
		return image

