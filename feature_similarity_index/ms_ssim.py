import numpy as np
from scipy.ndimage import zoom


def ssim(img1, img2):
	C1 = (0.01 * 255) ** 2
	C2 = (0.03 * 255) ** 2

	mu1 = np.mean(img1)
	mu2 = np.mean(img2)
	sigma1 = np.var(img1)
	sigma2 = np.var(img2)
	sigma12 = np.cov(img1.reshape(-1), img2.reshape(-1))[0, 1]

	ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) \
		/ ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

	return min(ssim_index, 1.0)


def lanczos_resample(image, scale_factor, a=3):
	"""
	Resample an image using the Lanczos filter.

	:param scale_factor: Scaling factor (less than 1 for downsampling, greater than 1 for upsampling).
	:param a: The size of the Lanczos window ('a' parameter).
	:return: Resampled image.
	"""
	return np.clip(zoom(image, zoom=scale_factor, order=a), 0, 255)


def ms_ssim(img1, img2):
	# Creating multiple scaled versions of the two images
	scale_factors = [1, 0.5, 0.25]  # Original, 50%, and 25% scales
	scaled_images1 = [lanczos_resample(img1, sf) for sf in scale_factors]
	scaled_images2 = [lanczos_resample(img2, sf) for sf in scale_factors]

	# Compute SSIM for each scale (comparing corresponding scaled versions of img1 and img2)
	ssim_values = [ssim(scaled_img1, scaled_img2) for scaled_img1, scaled_img2 in zip(scaled_images1, scaled_images2)]

	# Combining SSIM values across scales
	# Typically, the weight for the coarsest scale is higher
	weights = [0.5, 0.3, 0.2]  # Example weights, can be adjusted
	ms_ssim_value = np.prod([ssim_val ** weight for ssim_val, weight in zip(ssim_values, weights)])

	return ms_ssim_value


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

	scaled_image = lanczos_resample(image, 0.5)
	scaled_image = lanczos_resample(scaled_image, 2.0)

	ssim_index = ms_ssim(image, scaled_image)

	print(ssim_index)

	plt.subplot(1, 2, 1)
	plt.imshow(image, cmap='gray')
	plt.title('Original Image')
	annotate_image(image)
	plt.colorbar()

	plt.subplot(1, 2, 2)
	plt.imshow(scaled_image, cmap='gray')
	plt.title('Scaled Image')
	annotate_image(scaled_image)
	plt.colorbar()

	plt.tight_layout()
	plt.show()
