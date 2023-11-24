import numpy as np


def add_differential_encode(img):
	"""
	Differential encoding is a technique where, instead of storing the actual
	values, we store the difference between consecutive values in each row.
	This method can be useful in data compression and highlighting changes in
	the image data.

	Example:
	For the input array [[123, 125, 125], [1, 1, 6]], the function would work
	as follows:
	-	In the first row [123, 125, 125], the first value stays the same (123),
		and then we store the differences: 125 - 123 = 2, and 125 - 125 = 0.
		So, the first row in the output is [123, 2, 0].
	-	Similarly, in the second row [1, 1, 6], the first value is 1, and then
		the differences are 1 - 1 = 0, and 6 - 1 = 5. Hence, the second row in
		the output is [1, 0, 5].

	The resulting encoded array would be [[123, 2, 0], [1, 0, 5]].
	"""
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
	import matplotlib.pyplot as plt

	def annotate_image(image_array, font_size=8):
		for (j, i), value in np.ndenumerate(image_array):
			plt.text(i, j, f'{value:.0f}', ha='center', va='center', color='blue', fontsize=font_size)

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

	diff = add_differential_encode(np.array(image))
	print(diff)
	restore = add_differential_decode(np.array(diff))
	print(diff)

	plt.subplot(2, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	annotate_image(image)

	plt.subplot(2, 2, 2)
	plt.imshow(diff, cmap='gray')
	plt.title('Add difference Image')
	annotate_image(diff)

	plt.subplot(2, 2, 3)
	plt.imshow(restore, cmap='gray')
	plt.title('Restored Image')
	annotate_image(restore)

	plt.show()
