import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage import io, filters, morphology, measure

# Load the predicted image
predicted_image = io.imread('/home/jovyan/Desktop/shivanshi thesis/Tushar/mosaic1.tif', as_gray=True)

# Apply Otsu's thresholding to convert to binary image
threshold = filters.threshold_otsu(predicted_image)
binary_image = predicted_image > threshold

# Perform morphological operations (e.g., dilation, erosion)
dilated_image = morphology.dilation(binary_image)
eroded_image = morphology.erosion(dilated_image)

# Apply connected component analysis using OpenCV
_, labeled_image = cv2.connectedComponents(cv2.convertScaleAbs(eroded_image))

# Define canal type thresholds
thresholds = {
    'Main Canal': (38, 42),
    'Branch Canal': (18, 22),
    'Distributary': (13, 17),
    'Minors and Waterways': (3, 7)
}

# Create a new labeled image with canal types
classified_image = np.zeros_like(labeled_image, dtype=np.uint8)

# Iterate through labeled components and classify by width
for label in range(1, labeled_image.max() + 1):
    component = (labeled_image == label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        continue
    
    bbox = cv2.boundingRect(contours[0])
    width = bbox[2]
    
    canal_type = 'Other'
    for t, (min_width, max_width) in thresholds.items():
        if min_width < width < max_width:
            canal_type = t
            break
    
    classified_image[labeled_image == label] = canal_type

# Create a new TIF file
output_file = '/home/jovyan/Desktop/shivanshi thesis/Tushar/classified_image.tif'
height, width = classified_image.shape
transform = from_origin(0, 0, 1, 1)  # Define the transformation
dtype = classified_image.dtype

# Save the classified image
with rasterio.open(output_file, 'w', driver='GTiff', height=height, width=width, count=1, dtype=dtype, transform=transform) as dst:
    dst.write(classified_image, indexes=1)


