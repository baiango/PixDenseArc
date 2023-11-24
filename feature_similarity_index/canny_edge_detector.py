import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter


def non_max_suppression(gradient_magnitude, gradient_angle):
	"""
	Applies Non-Maximum Suppression to an image. This technique is used in edge
	detection to thin out the edges, making them sharp and clear. It works by
	comparing each pixel's gradient magnitude with its neighbors along the
	direction of the gradient angle and keeping only the local maxima (highest
	values).

	The function works as follows:
	1.	Start with a gradient magnitude image and its corresponding gradient
		angle.
	2.	Convert all gradient angles to degrees, making sure they are positive
		(0 to 180 degrees).
	3.	For each pixel in the image (ignoring the border pixels):
		a.	Determine its direction (0, 45, 90, or 135 degrees) based on the
			gradient angle.
		b.	Compare the pixel's gradient magnitude with the magnitudes of the
			two neighboring pixels in that direction.
		c.	If the pixel's magnitude is not the largest among the three, set it
			to 0 (suppress it).
	4.	The result is an image where only the local maxima (strongest edges)
		are kept, and all other pixels are set to 0.

	Parameters:
	gradient_magnitude (np.array): The gradient magnitudes of the image.
	gradient_angle (np.array): The gradient angles of the image, in radians.
	"""
	rows, cols = gradient_magnitude.shape
	suppressed_image = np.zeros((rows, cols), dtype=np.uint8)
	angle_degrees = gradient_angle * 180.0 / np.pi
	angle_degrees[angle_degrees < 0] += 180

	for row in range(1, rows - 1):
		for col in range(1, cols - 1):
			neighbor_1 = 255
			neighbor_2 = 255

			# Determine the direction and compare with the neighboring pixels
			if (0 <= angle_degrees[row, col] < 22.5) \
			or (157.5 <= angle_degrees[row, col] <= 180):
				neighbor_1 = gradient_magnitude[row, col + 1]
				neighbor_2 = gradient_magnitude[row, col - 1]
			elif (22.5 <= angle_degrees[row, col] < 67.5):
				neighbor_1 = gradient_magnitude[row + 1, col - 1]
				neighbor_2 = gradient_magnitude[row - 1, col + 1]
			elif (67.5 <= angle_degrees[row, col] < 112.5):
				neighbor_1 = gradient_magnitude[row + 1, col]
				neighbor_2 = gradient_magnitude[row - 1, col]
			elif (112.5 <= angle_degrees[row, col] < 157.5):
				neighbor_1 = gradient_magnitude[row - 1, col - 1]
				neighbor_2 = gradient_magnitude[row + 1, col + 1]

			# Keep only the local maxima
			if (gradient_magnitude[row, col] >= neighbor_1) \
			and (gradient_magnitude[row, col] >= neighbor_2):
				suppressed_image[row, col] = gradient_magnitude[row, col]
			else:
				suppressed_image[row, col] = 0

	return suppressed_image


def thresholding(edge_magnitude, low_threshold=8, high_threshold=15):
	"""
	Applies thresholding to an image based on edge magnitudes. This process
	helps in identifying and categorizing the strong and weak edges in the
	image.

	The function works as follows:
	1.	Initialize a result image of the same size as the input, initially with
		all values set to 0.
	2.	Define two types of edges: 'weak' and 'strong'.
	3.	Find the locations in the image where:
		a.	The edge magnitude is greater than or equal to the high threshold -
			mark these as 'strong' edges.
		b.	The edge magnitude is below the low threshold - these areas will
			remain 0, indicating no edge.
		c.	The edge magnitude is between the low and high thresholds - mark
			these as 'weak' edges.
	4.	The result is an image where edges are marked as either
		'strong', 'weak', or not an edge.

	Parameters:
	edge_magnitude (np.array): The array representing the magnitudes of edges in the image.

	In this process, 'strong' edges are considered significant, while 'weak'
	edges may be considered as potential edges but are not as prominent. This
	method is often used in edge detection algorithms to separate important
	edges from negligible ones.
	"""
	rows, cols = edge_magnitude.shape
	result_image = np.zeros((rows, cols), dtype=np.uint8)

	weak_value = np.uint8(25)
	strong_value = np.uint8(255)

	# Identifying strong, weak, and non-edges
	strong_edges_row, strong_edges_col = np.where(edge_magnitude >= high_threshold)
	weak_edges_row, weak_edges_col = np.where((edge_magnitude > low_threshold) & (edge_magnitude < high_threshold))

	# Marking strong and weak edges in the result image
	result_image[strong_edges_row, strong_edges_col] = strong_value
	result_image[weak_edges_row, weak_edges_col] = weak_value

	return result_image


def canny_edge_detector(img):
	"""
	This function goes through several steps to find the sharp edges in an
	image, like the borders of objects or text.

	The function works as follows:
	1.	Gaussian Filter: This is like a blur effect. It smooths the image to
		reduce noise and minor details, making the important edges more
		noticeable.
	2.	Sobel Filters: These are two special patterns (one for horizontal
		changes and one for vertical changes in the image) that help find
		edges. We slide these patterns over the image to measure changes in
		brightness or color.
	3.	Convolve: This is the process of sliding the Sobel Filters over the
		image and applying them to every part of the image to highlight the
		edges.
	4.	Sqrt (Square Root): After applying the Sobel Filters, we combine their
		results and take the square root. This step helps us measure the
		strength of the edges.
	5.	Arctan2: This math function helps us figure out the direction of the
		edges.
	6.	Non-Max Suppression: Imagine the edges as peaks on a graph. This step
		keeps the highest peaks (the strongest edges) and removes the smaller
		ones. It makes sure that the edges in the final image are thin and
		clear.
	7.	Thresholding: This is like setting a bar. Any edge that's strong enough
		to cross this bar is kept, while weaker edges are removed. This step
		finalizes which edges are important enough to show in the final image.

	These steps together help highlight the most significant edges in the
	image, making them clear and distinct, which is useful in many image
	processing tasks.
	"""
	blurred_image = gaussian_filter(img, sigma=2.5)

	sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	Ix = convolve(blurred_image, sobel_x)
	Iy = convolve(blurred_image, sobel_y)

	G = np.sqrt(Ix**2 + Iy**2)
	Theta = np.arctan2(Iy, Ix)

	nms_image = non_max_suppression(G, Theta)
	final_edges = thresholding(nms_image)
	return final_edges


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	def annotate_image(image_array, font_size=12):
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

	ced = canny_edge_detector(image)

	print(ced)

	plt.subplot(1, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	annotate_image(image)
	plt.colorbar()

	plt.subplot(1, 2, 2)
	plt.imshow(ced, cmap='gray')
	plt.title('Canny Edge Detector')
	annotate_image(ced)
	plt.colorbar()

	plt.tight_layout()
	plt.show()
