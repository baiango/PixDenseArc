import numpy as np


def karhunen_loeve_transform(image):
	"""
	Key Terms:
	-	Mean: This is like the average color or intensity for each column in
		the image. It's calculated to normalize the image, making sure we focus
		on the unique differences in the image rather than the overall average
		color or intensity.

	-	Covariance Matrix: Think of this as a report that tells us how
		different parts of the image are related to each other. It helps us
		understand the relationships between various parts of the image,
		showing if changes in one part (like getting brighter or darker) are
		related to changes in another part.

	-	Eigenvalues and Eigenvectors: These are mathematical tools that help us
		find the most interesting parts of the image.
		-	Eigenvalues tell us how much 'action' or variation is in each part
			of the image. Higher eigenvalues mean more important information.
		-	Eigenvectors point to the directions in the image where there's the
			most variation or change. They help us identify the most striking
			features or patterns in the image, like edges or unique colors.
	"""
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
	"""
	Calculates the mean and sorted eigenvectors of an image, which are
	essential components for image transformation and reconstruction.

	This function computes two key elements from the image data:
	1.	Mean: The average of each column in the image, representing the average
		value across all pixels.
	2.	Sorted Eigenvectors: These are calculated from the covariance matrix
		of the image and represent the principal components or the directions
		in which the image data varies the most.
	"""
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
	This function tweaks the finer, less noticeable parts of a pixel's value.
	The idea is to make these details more consistent across the image, which
	helps in compressing the image better.

	Here's a simpler way to understand what's happening:
	-	Think of each pixel value in the image as a series of switches (bits),
		where each switch can be either on or off. For example, the number 123
		in binary (switch) form is 01111011.
	-	We focus on the less noticeable switches (the rightmost bits) in each
		pixel's value.
	-	The 'value' you provide acts like a filter, deciding which of these
		switches we should pay attention to.
	-	If we use 15 (0xf, or 00001111 in binary) as the value, it means we're
		focusing on the last 4 bits of the pixel's value. So, 123 becomes 11
		(00001011 in binary).
	-	Depending on the original shade of the pixel (lighter or darker),
		we then adjust the pixel's value slightly by adding or subtracting
		this new number.

	Parameters:
	img (np.array):	The input image array. Imagine this as a collection of
					pixels, where each pixel is represented by a number.
	value (int):	A number that acts as a filter for the pixel's value.
					Typical values include:
					-	0xf (15 in decimal): Ignores the first 4 bits.
					-	Other values like 0x1, 0x3, 0x7, 0x1f (31), 0x3f (63),
						0x7f (127), and 0xff (255) can also be used,
						each focusing on different parts of the pixel's value.
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
