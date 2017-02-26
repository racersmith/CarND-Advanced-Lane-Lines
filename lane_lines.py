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

		# Show perspective points
		compare_image(image1, "Source", image2, "Destination", axii='on')

	def warp(self, image):
		""" Warp image to bird's eye view """
		return cv2.warpPerspective(image, self.M, self.image_size, flags=cv2.INTER_NEAREST)

	def unwarp(self, image):
		""" Unwarp image from bird's eye view to driver's view """
		return cv2.warpPerspective(image, self.Minv, self.image_size, flags=cv2.INTER_NEAREST)

class LaneLines:
	def __init__(self, image_size, pixels_per_meter=(24, 189), camera):
		self.image_size = image_size
		self.pixels_per_meter = pixels_per_meter
		self.ym_per_pix = 1/pixels_per_meter[0]
		self.xm_per_pix = 1/pixels_per_meter[1]
		self.left_mask = None
		self.right_mask = None
		self.left_fit = None
		self.right_fit = None
		self.left_fitx = None
		self.right_fitx = None
		self.ploty = None
		self.leftx = None
		self.lefty = None
		self.rightx = None
		self.righty = None
		self.left_curverad = None
		self.right_curverad = None
		self.vehicle_offset = None
		self.camera = camera

	def binary_threshold(self, image, thresh=(0, 255)):
		"""Convert the image to binary based on the pixle value falling
		with in the threshold boundary"""
		# Rescale to 8 bit integer
		scaled_image = np.uint8(255*image/np.max(image))
		# Create a copy and apply the threshold
		binary_output = np.zeros_like(scaled_image)
		# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
		binary_output[(scaled_image >= thresh[0]) & (scaled_image <= thresh[1])] = 1
		return binary_output

	def sobel(self, image, orient='x', sobel_kernel=3):
		"""Gradient in either the x or y direction"""
		if orient == 'x':
			sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		else:
			sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		return sobel

	def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
		"""Absolute value of the gradient in either the x or y direction"""
		# Take gradient in specified direction
		gradient = self.sobel(image, orient, sobel_kernel)
		# absolute value of gradient
		abs_sobel = np.absolute(gradient)
		return self.binary_threshold(abs_sobel, thresh)

	def clean_binary(self, image, close_kernel, open_kernel):
		""" close holes in image then remove/open small areas
		image -- binary image to clean
		close_kernel -- (height, width) of kernel to use for close
		open_kernel --(height, width) of kernel to use for open
		"""
		kernel = np.ones(close_kernel,np.uint8)
		image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
		kernel = np.ones(open_kernel,np.uint8)
		image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
		return image    

	def binary_image(self, image):
		image = np.copy(image)

		# convert to HLS color space
		hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		l_channel = hls[:,:,1]
		s_channel = hls[:,:,2]
		s_eq = cv2.equalizeHist(s_channel).astype(np.float)

		# Extract potential lane points
		binary_lx = self.abs_sobel_thresh(l_channel, 'x', thresh=(40, 255))
		binary_lx = self.clean_binary(binary_lx, (15,15), (11,5))
		binary_sx = self.abs_sobel_thresh(s_channel, 'x', thresh=(50, 255))    
		binary_sx = self.clean_binary(binary_sx, (5,5), (11,5))
		binary_s = self.binary_threshold(s_eq, thresh=(245, 255))
		binary_s = self.clean_binary(binary_s, (3,3), (11,5))
		combined = np.zeros_like(l_channel)    
		combined[(binary_lx==1) | (binary_sx==1) | (binary_s==1)] = 1
		return combined

	def find_lost_line_points(self, binary_warped):
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(binary_warped.shape[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		    if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		self.leftx = nonzerox[left_lane_inds]
		self.lefty = nonzeroy[left_lane_inds] 
		self.rightx = nonzerox[right_lane_inds]
		self.righty = nonzeroy[right_lane_inds]

	def find_line_points(self, binary_warped):
		# Identify the x and y positions of all nonzero pixels in the image
		left_points = binary_warped & self.left_mask
		
		nonzero = left_points.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		self.leftx = nonzerox[left_lane_inds]
		self.lefty = nonzeroy[left_lane_inds] 
		
		right_points = binary_warped & self.right_mask
		nonzero = right_points.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		self.rightx = nonzerox[right_lane_inds]
		self.righty = nonzeroy[right_lane_inds]

	def line_mask(self, margin):
		# create image to create mask
		self.left_mask = np.zeros(self.image_size)
		self.right_mask = np.zeros(self.image_size)
		self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
		self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
		self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, self.ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, self.ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Apply mask region
		cv2.fillPoly(self.left_mask, np.int_([left_line_pts]), 1)
		cv2.fillPoly(self.right_mask, np.int_([right_line_pts]), 1)

	def fit_lines(self):
		# Fit a second order polynomial to each
		self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
		self.right_fit = np.polyfit(self.righty, self.rightx, 2)

	def lane_stats(self):
		# Define conversions in x and y from pixels space to meters
		self.ym_per_pix = 30/720 # meters per pixel in y dimension
		self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
		y_eval = 720
		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)
		# Calculate the new radii of curvature
		self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		# Determine vehicle offset from center
		left_pos = self.left_fit[0]*y_eval**2 + self.left_fit[1]*y_eval + self.left_fit[2]
		right_pos = self.right_fit[0]*y_eval**2 + self.right_fit[1]*y_eval + self.right_fit[2]
		lane_center = (left_pos + right_pos)/2*self.xm_per_pix
		vehicle_offset = image_size[0]*self.xm_per_pix/2 - lane_center

	def mark_lane(self, image):
		image = camera.undistort()
		image = camera.warp()
		bin_image = self.binary_image(image)
		
		if not self.left_mask | not self.right_mask:
			self.find_lost_line_points(bin_image)
		else:
			self.find_line_points(bin_image)
		self.fit_lines()
		self.line_mask(100)
		self.lane_stats()
		
		if max(self.left_curverad, self.right_curverad)/min(self.left_curverad, self.right_curverad) > 10:
			self.left_max, self.right_mask = None

		# Create an image to draw the lines on
		warp_zero = np.zeros_like(bin_image).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = camera.unwarp(color_warp)
		# Combine the result with the original image
		result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
		plt.imshow(result)
		plt.show()




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