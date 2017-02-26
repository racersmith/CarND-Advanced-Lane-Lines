import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math

class Camera:
	def __init__(self):
		self.calibrated = False
		self.mtx = None
		self.dist = None
		self.good_cal_images = 0
		self.M = None
		self.Minv = None
		self.image_size = None

	def calibrate(self, chessboard=(9, 6), folder='camera_cal/', verbose = False):
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((np.product(chessboard),3), np.float32)
		objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d points in real world space
		imgpoints = [] # 2d points in image plane.

		# Make a list of calibration images
		images = glob.glob(folder+'calibration*.jpg')

		# Step through the list and search for chessboard corners
		for idx, fname in enumerate(images):
			if verbose:
				print("Processing image %i\t" % (idx+1), end='')
			image = cv2.imread(fname)
			if verbose:
				print("  Image imported", end='')
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			if verbose:
				print("  Converted to grayscale", end='')

			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray, chessboard, None)

			# If found, add object points, image points
			if ret == True:
				self.good_cal_images += 1
				if verbose:
					print("  Corners found!")
				objpoints.append(objp)
				imgpoints.append(corners)

				# Draw and display the corners
				if verbose:
					cv2.drawChessboardCorners(image, chessboard, corners, ret)
					cv2.imshow('image', image)
					cv2.waitKey(500)
			else:
				if verbose:
					print("  No corners found :(")
		print('Calibration complete. %i good calibaration images out of %i total.' % (self.good_cal_images, len(images)))
		cv2.destroyAllWindows()

		# Do camera calibration given object points and image points
		self.image_size = (image.shape[1], image.shape[0])
		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.image_size,None,None)
		self.calibrated = True

	def undistort(self, image):
		""" Correct for camera distortion
		image -- image to undistort
		returns -- undistorted image
		"""
		return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

	def set_perspective_points(self, src, dst, image):
		# calculate and save perspective transform matricies
		self.M = cv2.getPerspectiveTransform(src,dst)
		self.M_inv = cv2.getPerspectiveTransform(dst,src)

		# Draw source points on unwarped image
		image1 = self.undistort(image)
		src_draw = src.reshape((-1,1,2)).astype(int)
		cv2.polylines(image1, [src_draw], True, (98, 244, 65), thickness=4)

		# Draw destination points on warped image
		image2 = self.undistort(image)
		image2 = self.warp(image2)
		dst_draw = dst.reshape((-1,1,2)).astype(int)
		cv2.polylines(image2, [dst_draw], True, (98, 244, 65), thickness=4)
		compare_image(image1, "Source", image2, "Destination", axii='on')

	def warp(self, image):
		return cv2.warpPerspective(image, self.M, self.image_size, flags=cv2.INTER_NEAREST)

	def unwarp(self, image):
		return cv2.warpPerspective(image, self.Minv, self.image_size, flags=cv2.INTER_NEAREST)

def compare_image(image1, title1, image2, title2, axii='off'):
	""" plot two images side by side for comparison
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.imshow(image1)
	ax1.set_title(title1, fontsize=30)
	ax1.axis(axii)
	ax2.imshow(image2)
	ax2.set_title(title2, fontsize=30)
	ax2.axis(axii)
	plt.show()