import numpy as np
from scipy.ndimage import zoom


def ssim(img1, img2):
	"""
	Structural Similarity Index (SSIM) is a measure used to quantify the
	similarity between two images. It is more perceptually accurate than simple
	difference measures, as it incorporates important aspects of human visual
	perception.

	Why C1 and C2 are there:
	-	C1 and C2 are small constants added to stabilize the division with weak
		denominators. They prevent division by zero when the mean or variance
		values are very small. C1 is based on 0.01^2 of the dynamic range of
		pixel values (here assumed to be 255), and C2 is based on 0.03^2 of the
		same range.

	np.mean:
	-	This function calculates the average pixel value of the images. It's
		used to estimate the luminance (brightness) of each image, which is a
		crucial component in the SSIM index.

	np.var:
	-	This function calculates the variance of the pixel values in each
		image. Variance measures how spread out the values are. In SSIM, it
		helps in assessing the contrast or texture of the images.

	np.cov:
	-	This function calculates the covariance between the two images.
		Covariance measures how much the images change together, which helps in
		evaluating the structural similarity. In SSIM, it's used to compare the
		pattern of variation between the two images.

	The SSIM index combines these measurements (luminance, contrast, and
	structure) into a single value that ranges between -1 and 1, where 1
	indicates perfect similarity.

	Parameters:
	img1, img2 (np.array): The input images to compare.

	Note: The SSIM index is clipped at 1 to handle edge cases where it might
	exceed 1 due to numerical errors.
	"""
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
	The Lanczos filter is a type of windowed sinc filter that is used for
	resizing images. It works by considering a larger area around each pixel to
	decide its new value after resizing. This makes it effective at preserving
	the detail and texture of the original image, especially compared to
	simpler methods like nearest-neighbor or bilinear interpolation.

	Here's a brief on how it works:
	-	Scale Factor: Determines how much the image should be resized. A scale
		factor less than 1 will make the image smaller (downsampling), and a
		factor greater than 1 will make it larger (upsampling).
	-	'a' Parameter: Defines the size of the window that the Lanczos filter
		considers around each pixel. A larger 'a' results in more neighboring
		pixels being considered, which can lead to better preservation of
		detail but may introduce more computation.
	"""
	return np.clip(zoom(image, zoom=scale_factor, order=a), 0, 255)


def ms_ssim(img1, img2):
	"""
	Multi-Scale Structural Similarity Index (MS-SSIM) is a method to measure
	how similar two images are, not just by looking at them at one scale, but
	by comparing them at multiple scales (sizes). This helps in understanding
	the similarity in structure, brightness, and contrast at different levels
	of detail.

	Here's what the function does:
	1.	Create Scaled Versions: It first creates smaller versions of the
		original images. For example, if we start with an image, we create
		half-sized (50%) and quarter-sized (25%) versions of the same.
	2.	Compare at Each Scale: For each scale, it uses a method called SSIM
		(Structural Similarity Index) to compare how similar the two images are
		at that scale. SSIM is a way to measure the similarity in terms of
		structure, brightness, and contrast.
	3.	Combine Results: Finally, it combines these similarity measurements
		from all scales into a single score. This score tells us how similar
		the two images are overall, considering different levels of detail.
	"""
	# Creating multiple scaled versions of the two images
	scale_factors = [1, 0.5, 0.25]  # Original, 50%, and 25% scales
	scaled_images1 = [lanczos_resample(img1, sf) for sf in scale_factors]
	scaled_images2 = [lanczos_resample(img2, sf) for sf in scale_factors]

	# Compute SSIM for each scale (comparing corresponding scaled versions of img1 and img2)
	ssim_values = [ssim(scaled_img1, scaled_img2) for scaled_img1, scaled_img2 in zip(scaled_images1, scaled_images2)]

	# Combining SSIM values across scales
	# Typically, the weight for the coarsest scale is higher
	score_weights = [0.5, 0.3, 0.2]  # Can be adjusted
	ms_ssim_value = np.prod([ssim_val ** weight for ssim_val, weight in zip(ssim_values, score_weights)])

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
