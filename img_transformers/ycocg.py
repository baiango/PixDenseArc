import numpy as np


def rgb_to_ycocg(rgb):
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
