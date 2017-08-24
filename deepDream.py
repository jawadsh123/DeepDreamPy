import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import math
import random

import inception5h


model = inception5h.Inception5h()

# calcuating gradient equation
LAYER_INDEX = 7
layer_tensor = model.layer_tensors[LAYER_INDEX]
gradient = model.get_gradient(layer_tensor)


# starting the session
session = tf.InteractiveSession(graph=model.graph)


def load_image(fileName):
	fileName = "./images/" + fileName
	image = Image.open(fileName)
	return np.float32(image)

def save_image(image, fileName, replace_original=False):
	global LAYER_INDEX

	if not replace_original:
		fileName += "_dream_"+str(LAYER_INDEX)+".jpg"
	else:
		fileName += "_"+str(LAYER_INDEX)+".jpg"

	fileName = "./images/" + fileName
	image = np.clip(a=image, a_min=0.0, a_max=255.0)
	image = image.astype(np.uint8)

	with open(fileName, "wb") as file:
		Image.fromarray(image).save(file, "jpeg")

def compute_tile_size(total_pixels, tile_size=400):

	num_tiles = int(round(total_pixels/tile_size))
	num_tiles = max(1, num_tiles)
	actual_tile_size = math.ceil(total_pixels/num_tiles)

	return actual_tile_size

def resize_image(image, size=None, factor=None):

	if factor is not None:
		size = np.array(image.shape[0:2]) * factor
		size = size.astype(int)
	else:
		size = size[0:2]

	# since height an width is reversed in numpy and PIL
	size = tuple(reversed(size))

	img = np.clip(image, 0.0, 255.0)
	img = img.astype(np.uint8)

	img = Image.fromarray(img)
	img_resized = img.resize(size, Image.LANCZOS)
	img_resized = np.float32(img_resized)

	return img_resized


def compute_gradient(image, tile_size=400):
	global layer_tensor, gradient

	grad = np.zeros_like(image)

	x_max, y_max, _ = image.shape

	x_tile_size = compute_tile_size(total_pixels=x_max, tile_size=tile_size)
	x_tile_size_4th = x_tile_size // 4

	y_tile_size = compute_tile_size(total_pixels=y_max, tile_size=tile_size)
	y_tile_size_4th = y_tile_size // 4

	x_start = random.randint(-3*x_tile_size_4th, -x_tile_size_4th)

	while x_start < x_max:
		x_end = x_start + x_tile_size

		x_start = max(0, x_start)
		x_end = min(x_max, x_end)

		y_start = random.randint(-3*y_tile_size_4th, -y_tile_size_4th)
		while y_start < y_max:
			y_end = y_start + y_tile_size

			y_start = max(0, y_start)
			y_end = min(y_max, y_end)

			tile = image[x_start:x_end, y_start:y_end, :]

			feed_dict = model.create_feed_dict(image=tile)
			tiled_grad = session.run(gradient, feed_dict=feed_dict)
			tiled_grad = tiled_grad / (np.std(tiled_grad) + 1e-8)

			grad[x_start:x_end, y_start:y_end, :] = tiled_grad

			y_start = y_end
		x_start = x_end

	return grad

def optimize_image(image, num_iterations, step_size=3.0, tile_size=400):

	img = image.copy()

	for i in range(num_iterations):
		grad = compute_gradient(img, tile_size=tile_size)

		# sigma = (i * 4.0) / num_iterations + 0.5
		# grad_smooth1 = gaussian_filter(grad, sigma=sigma)
		# grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
		# grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
		# grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

		sigma = (i * 4.0) / num_iterations + 0.5
		grad_smooth1 = gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
		grad_smooth2 = gaussian_filter(grad, sigma=(sigma*2, sigma*2, 0.0))
		grad_smooth3 = gaussian_filter(grad, sigma=(sigma*0.5, sigma*0.5, 0.0))
		grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

		step_size_normalized = step_size / (np.std(grad) + 1e-8)
		img += step_size_normalized * grad

		print("Optimization Iteration: {}".format(i))
		print("Max Gradient: {}, Min: {}, Step Size: {}".format(np.max(grad), np.min(grad), step_size_normalized))

	return img

def recursively_optimize_image(num_repeats, image, num_iterations, step_size=3.0, tile_size=400, blend=0.2, rescale_factor=0.7):

	if num_repeats > 0:
		# amount of blur
		sigma = 0.5
		img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

		downscaled_image = resize_image(image=img_blur, factor=rescale_factor)

		dreamified_image = recursively_optimize_image(num_repeats=num_repeats-1, image=downscaled_image, \
			num_iterations=num_iterations, step_size=step_size, tile_size=tile_size, blend=blend, rescale_factor=rescale_factor)

		upscaled_image = resize_image(dreamified_image, size=image.shape)

		image = blend * image + (1-blend) * upscaled_image

	print("============================")
	print("Depth: {}".format(num_repeats))

	dreamified_image = optimize_image(image=image, num_iterations=num_iterations, step_size=step_size, tile_size=tile_size)

	return dreamified_image

# def recursively_optimize_for_every_layer(num_repeats, image, num_iterations, step_size=3.0, tile_size=400, blend=0.2, rescale_factor=0.7):

# 	layers = model.layer_tensors

# 	for layer_no in range(len(layers)):

# 		changed_image = recursively_optimize_image(num_repeats=4, image=image, num_iterations=10, step_size=3.0, tile_size=400, blend=0.2, rescale_factor=0.7)




INPUT_IMAGE = "landscape"
image = load_image(INPUT_IMAGE+".jpg")
# changed_image = optimize_image(image=image, num_iterations=100, step_size=3.0)
changed_image = recursively_optimize_image(num_repeats=10, image=image, num_iterations=10, step_size=3.0, tile_size=512, blend=0.2, rescale_factor=0.7)
save_image(image=changed_image, fileName=INPUT_IMAGE, replace_original=True)