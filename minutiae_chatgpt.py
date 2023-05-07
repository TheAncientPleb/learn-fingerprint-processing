import cv2
import numpy as np

# Load the fingerprint image
img = cv2.imread('fingerprint.png', 0)

# Apply a threshold to the image
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

# Define the structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Perform morphological opening to remove noise
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Find contours in the image
contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Define an empty list to store minutiae
minutiae = []

# Loop through the contours
for cnt in contours:
    # Define an empty list to store branch and endpoint minutiae for this contour
    branch_pts = []
    end_pts = []
    
    # Get the convex hull of the contour
    hull = cv2.convexHull(cnt)
    
    # Loop through the hull points
    for i in range(len(hull)):
        # Get the coordinates of the current point
        p1 = hull[i][0]
        
        # Check if the current point is a branch or an endpoint
        if i == 0:
            p2 = hull[-1][0]
            p3 = hull[1][0]
        elif i == len(hull)-1:
            p2 = hull[-2][0]
            p3 = hull[0][0]
        else:
            p2 = hull[i-1][0]
            p3 = hull[i+1][0]
        
        # Calculate the angles between the current point and its neighbors
        angle1 = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        angle2 = np.arctan2(p3[1]-p1[1], p3[0]-p1[0])
        angle = np.abs(angle1-angle2)
        
        # If the angle is less than 90 degrees, the current point is a branch
        if angle < np.pi/2:
            branch_pts.append(p1)
        # If the angle is greater than 120 degrees, the current point is an endpoint
        elif angle > 2*np.pi/3:
            end_pts.append(p1)
    
    # Add the branch and endpoint minutiae for this contour to the list of minutiae
    for pt in branch_pts:
        minutiae.append({'type': 'branch', 'x': pt[0], 'y': pt[1]})
    for pt in end_pts:
        minutiae.append({'type': 'endpoint',
