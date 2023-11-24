import numpy as np


def add_differential_encode(img):
	encoded_img = np.zeros_like(img, dtype=np.uint8)
	x, y = img.shape
	for i in range(x):
		encoded_img[i, 0] = img[i, 0]  # Keep the first column unchanged
		for j in range(1, y):
			encoded_img[i, j] = (img[i, j] - img[i, j - 1])
	return encoded_img


def add_differential_decode(encoded_img):
	restored_img = np.zeros_like(encoded_img, dtype=np.uint8)
	x, y = encoded_img.shape
	for i in range(x):
		restored_img[i, 0] = encoded_img[i, 0]  # Keep the first column unchanged
		for j in range(1, y):
			restored_img[i, j] = (restored_img[i, j - 1] + encoded_img[i, j])
	return restored_img


if __name__ == '__main__':
	image = np.array([
		[0.0, 0.0, 0.0, 255.0, 0.0, 50.0, 0.0, 255.0],
		[0.0, 0.0, 0.0, 0.0, 50.0, 255.0, 50.0, 0.0],
		[0.0, 0.0, 0.0, 50.0, 255.0, 255.0, 255.0, 50.0],
		[0.0, 0.0, 0.0, 0.0, 50.0, 255.0, 50.0, 0.0],
		[0.0, 255.0, 255.0, 0.0, 0.0, 50.0, 0.0, 255.0],
		[255.0, 100.0, 100.0, 255.0, 0.0, 0.0, 0.0, 0.0],
		[255.0, 100.0, 100.0, 255.0, 0.0, 0.0, 0.0, 0.0],
		[255.0, 100.0, 100.0, 255.0, 0.0, 0.0, 0.0, 0.0],
	])

	diff = add_differential_encode(image)
	print(diff)
	diff = add_differential_decode(diff)
	print(diff)
