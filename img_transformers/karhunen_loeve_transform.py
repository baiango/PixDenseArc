import numpy as np


def karhunen_loeve_transform(image):
	# Calculate the mean of the image
	mean = np.mean(image, axis=0)

	# Center the image by subtracting the mean
	centered_image = image - mean

	# Calculate the covariance matrix
	covariance_matrix = np.cov(centered_image, rowvar=False)

	# Perform eigenvalue decomposition on the covariance matrix
	eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

	# Sort the eigenvalues and eigenvectors
	sorted_indices = np.argsort(eigenvalues)[::-1]
	sorted_eigenvectors = eigenvectors[:, sorted_indices]

	# Transform the image using the sorted eigenvectors
	transformed_image = np.dot(centered_image, sorted_eigenvectors)

	return transformed_image


def generate_mean_and_eigenvectors(img):
	mean = np.mean(img, axis=0)
	covariance_matrix = np.cov(img, rowvar=False)
	eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
	sorted_indices = np.argsort(eigenvalues)[::-1]
	sorted_eigenvectors = eigenvectors[:, sorted_indices]
	return mean, sorted_eigenvectors


def inverse_karhunen_loeve_transform(transformed_image, mean, eigenvectors):
	# Reconstruct the original image using the eigenvectors
	reconstructed_image = np.dot(transformed_image, eigenvectors.T)

	# Add the mean back to the reconstructed image
	reconstructed_image += mean

	return reconstructed_image


def remap(a, a_min, a_max, b_min, b_max):
	return (a - a_min) * (b_max - b_min) / (a_max - a_min) + b_min


def bitwise_and_sign_subtract(img, value):
	"""
	Conditionally adjusts each pixel in an image for compression purposes by manipulating its least significant bits.

	This function targets the least significant bits of pixel values. By converting each pixel
	value to its absolute value, applying a bitwise AND with 'value', and then conditionally
	subtracting or adding this result based on the original sign (positive or negative), the
	function can effectively reduce randomness in these lower bits. This is useful in image
	compression scenarios where reducing randomness or 'noise' in the least significant bits
	can lead to better compression efficiency.

	The function operates by first converting each pixel value to its absolute value. Then, it applies a bitwise
	AND operation with the specified value. Depending on whether the original pixel value was positive or negative,
	it either subtracts or adds the result of this bitwise operation to the original pixel value.

	Parameters:
	img (np.array): The input image array.
	value (int): A bitwise mask used for the AND operation. Typical values for reducing randomness in
				 compression include 0xf (ignores the first 4 bits), as well as other hexadecimal values
				 like 0x1, 0x3, 0x7, 0x1f (31), 0x3f (63), 0x7f (127), and 0xff (255).
	"""
	img = img.astype(np.int64)
	sign_mask = np.sign(img)
	abs_modified = np.abs(img) & value
	return img - sign_mask * abs_modified


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	def annotate_image(image_array, font_size=8):
		for (j, i), value in np.ndenumerate(image_array):
			plt.text(i, j, f'{value:.0f}', ha='center', va='center', color='blue', fontsize=font_size)

	# Define the image
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

	# rows, cols = 8, 8
	# increment_value = 4.0
	# image = np.arange(0, rows * cols * increment_value, increment_value).reshape(rows, cols)

	# image = np.zeros((rows, cols), dtype=np.uint8)
	# image.fill(255)

	# Flip the array vertically using numpy.flipud()
	image = np.flipud(image)
	# Flip the array horizontally using numpy.fliplr()
	image = np.fliplr(image)


	# Apply the Karhunen-Loeve Transform
	transformed_image = karhunen_loeve_transform(image)

	modified_image = np.copy(transformed_image)
	modified_image = bitwise_and_sign_subtract(modified_image, 0x7)

	# The mean and eigenvectors used for the KLT need to be stored for the inverse transform
	mean, sorted_eigenvectors = generate_mean_and_eigenvectors(image)
	scaled_eigenvectors = [remap(x, -1.0, 1.0, 0.0, 255.0) for x in sorted_eigenvectors]
	reconstructed_image = inverse_karhunen_loeve_transform(modified_image, mean, sorted_eigenvectors)

	plt.subplot(3, 3, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	annotate_image(image)
	plt.colorbar()

	plt.subplot(3, 3, 2)
	plt.imshow(transformed_image, cmap='gray')
	plt.title('Karhunen-Loeve Transform')
	annotate_image(transformed_image)
	plt.colorbar()

	plt.subplot(3, 3, 3)
	plt.imshow(modified_image, cmap='gray')
	plt.title('Modified Image')
	annotate_image(modified_image)
	plt.colorbar()

	prt = np.zeros((1, 8)) + mean
	plt.subplot(3, 3, 4)
	plt.imshow(prt, cmap='gray')
	plt.title('Mean')
	annotate_image(prt)
	plt.colorbar()

	plt.subplot(3, 3, 5)
	plt.imshow(sorted_eigenvectors, cmap='gray')
	plt.title('Sorted Eigenvectors')
	annotate_image(sorted_eigenvectors)
	plt.colorbar()

	plt.subplot(3, 3, 6)
	plt.imshow(scaled_eigenvectors, cmap='gray')
	plt.title('Scaled Eigenvectors')
	annotate_image(scaled_eigenvectors)
	plt.colorbar()

	plt.subplot(3, 3, 7)
	plt.imshow(reconstructed_image, cmap='gray')
	plt.title('Reconstructed Image')
	annotate_image(reconstructed_image)
	plt.colorbar()

	plt.tight_layout()
	plt.show()
