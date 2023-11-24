import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter


def non_max_suppression(G, Theta):
	M, N = G.shape
	Z = np.zeros((M,N), dtype=np.uint8)
	angle = Theta * 180. / np.pi
	angle[angle < 0] += 180

	for i in range(1, M-1):
		for j in range(1, N-1):
			q = 255
			r = 255

			# angle 0
			if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
				q = G[i, j+1]
				r = G[i, j-1]
			# angle 45
			elif (22.5 <= angle[i,j] < 67.5):
				q = G[i+1, j-1]
				r = G[i-1, j+1]
			# angle 90
			elif (67.5 <= angle[i,j] < 112.5):
				q = G[i+1, j]
				r = G[i-1, j]
			# angle 135
			elif (112.5 <= angle[i,j] < 157.5):
				q = G[i-1, j-1]
				r = G[i+1, j+1]

			if (G[i,j] >= q) and (G[i,j] >= r):
				Z[i,j] = G[i,j]
			else:
				Z[i,j] = 0

	return Z


def thresholding(H, lowThreshold=8, highThreshold=15):
	M, N = H.shape
	res = np.zeros((M,N), dtype=np.uint8)

	weak = np.uint8(25)
	strong = np.uint8(255)

	strong_i, strong_j = np.where(H >= highThreshold)
	zeros_i, zeros_j = np.where(H < lowThreshold)

	weak_i, weak_j = np.where((H > lowThreshold) & (H < highThreshold))

	res[strong_i, strong_j] = strong
	res[weak_i, weak_j] = weak

	return res


def canny_edge_detector(img):
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
