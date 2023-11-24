from canny_edge_detector import canny_edge_detector
from ms_ssim import ms_ssim
import numpy as np
from scipy.ndimage import gaussian_filter

def fsim(gray1, gray2):
	edges_img1 = canny_edge_detector(gray1)
	edges_img2 = canny_edge_detector(gray2)

	return ms_ssim(edges_img1, edges_img2)

def fsim_image(img1, img2):
	fsim_score = 0

	# Split each image into R, G, B channels
	for i in range(3):
		channel_img1 = img1[:, :, i]
		channel_img2 = img2[:, :, i]

		fsim_score += fsim(channel_img1, channel_img2)

	return fsim_score / 3.0


if __name__ == "__main__":
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

	scaled_image = gaussian_filter(image, 0.5)

	edges_original = canny_edge_detector(image)
	edges_scaled = canny_edge_detector(scaled_image)

	fsim_score = fsim(image, scaled_image)

	print(fsim_score)

	plt.subplot(2, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	annotate_image(image)
	plt.colorbar()

	plt.subplot(2, 2, 2)
	plt.imshow(scaled_image, cmap='gray')
	plt.title('Scaled Image')
	annotate_image(scaled_image)
	plt.colorbar()

	plt.subplot(2, 2, 3)
	plt.imshow(edges_original, cmap='gray')
	plt.title('Original Edges')
	annotate_image(edges_original)
	plt.colorbar()

	plt.subplot(2, 2, 4)
	plt.imshow(edges_scaled, cmap='gray')
	plt.title('Scaled Edges')
	annotate_image(edges_scaled)
	plt.colorbar()

	plt.tight_layout()
	plt.show()
