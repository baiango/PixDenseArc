import numpy as np


def rgb_to_ycocg(rgb):
	"""
	Convert a color from RGB (Red, Green, Blue) space to YCoCg
	(Luma, Orange Chroma, Green Chroma) space.

	YCoCg is a color space where 'Y' shows how bright the color is, 'Co' shows
	the difference between red and blue, and 'Cg' shows the difference between
	green and the average of red and blue.

	Note on Co and Cg:
	-	'Co' (Orange Chroma) is found by looking at the difference between red
		and blue. Changes in 'Co' are not as easily seen by our eyes compared
		to changes in 'Y' (brightness).
	-	'Cg' (Green Chroma) is found by looking at the difference between green
		and the average of red and blue. Like 'Co', changes in 'Cg' are also
		less noticeable compared to changes in brightness.

	This is useful in making images and videos smaller (compression) because we
	can reduce the amount of color information (Co and Cg) more than the
	brightness information (Y) without making a big difference to how the image
	looks to us.

	Works Well with Chroma Subsampling 4:2:0:
	-	YCoCg is great for a type of video compression called chroma
		subsampling 4:2:0. In this method, we reduce the color information more
		than the brightness information. Since we don't easily notice changes
		in color detail, this makes the image or video smaller without making
		it look much different, which helps in saving space and making files
		smaller.
	"""
	r, g, b = rgb[0], rgb[1], rgb[2]

	co = r - b
	tmp = b + co / 2
	cg = g - tmp
	y = tmp + cg / 2

	return [y, co, cg]

def ycocg_to_rgb(ycocg):
	y, co, cg = ycocg[0], ycocg[1], ycocg[2]

	tmp = y - cg / 2
	g = cg + tmp
	b = tmp - co / 2
	r = b + co

	return [r, g, b]

def rgb_to_ycocg_2d(image):
	# Convert the PIL Image to a NumPy array
	rgb_array = np.array(image)
	# Prepare an array for the YCoCg result
	ycocg_array = np.zeros_like(rgb_array, dtype=float)

	# Loop over each pixel
	for i in range(rgb_array.shape[0]):
		for j in range(rgb_array.shape[1]):
			# Apply the rgb_to_ycocg transformation
			ycocg_array[i, j] = rgb_to_ycocg(rgb_array[i, j])

	return ycocg_array

def ycocg_to_rgb_2d(image):
	# Convert the PIL Image to a NumPy array
	ycocg_array = np.array(image)
	# Prepare an array for the RGB result
	rgb_array = np.zeros_like(ycocg_array, dtype=float)

	# Loop over each pixel
	for i in range(ycocg_array.shape[0]):
		for j in range(ycocg_array.shape[1]):
			# Apply the ycocg_to_rgb transformation
			rgb_array[i, j] = ycocg_to_rgb(ycocg_array[i, j])

	return rgb_array

if __name__ == '__main__':
	rgb = [255, 0, 0]
	ycocg = rgb_to_ycocg(rgb)
	reconstructed_rgb = ycocg_to_rgb(ycocg)

	print("RGB:", rgb)
	print("RGB to YCoCg:", ycocg)
	print("YCoCg to RGB reconstructed:", reconstructed_rgb)
