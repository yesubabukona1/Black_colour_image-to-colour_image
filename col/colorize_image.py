import cv2
import numpy as np
import os

# Paths to model files
proto_path = r"models/colorization_deploy_v2.prototxt"
model_path = r"models/colorization_release_v2.caffemodel"
points_path = r"models/pts_in_hull.npy"

# Load model
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
pts_in_hull = np.load(points_path)
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

# Assign cluster centers as layers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts_in_hull.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Get user input for image path
image_path = input("Enter the path of the grayscale image: ").strip()

# Validate the image file
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found. Check the path.")
    exit()

# Load grayscale image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to read the image.")
    exit()

# Convert to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Extract L channel and resize
L = lab_image[:, :, 0]
L_resized = cv2.resize(L, (224, 224))
L_resized = L_resized.astype("float32") - 50  # Mean-centering

# Prepare input for model
net.setInput(cv2.dnn.blobFromImage(L_resized))

# Predict AB channels
ab_output = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize AB channels to match original image size
ab_output = cv2.resize(ab_output, (image.shape[1], image.shape[0]))

# Merge L with predicted AB channels
lab_colorized = np.zeros_like(lab_image)
lab_colorized[:, :, 0] = L  # Use original L channel
lab_colorized[:, :, 1:] = ab_output  # Add predicted AB

# Convert LAB to BGR
colorized = cv2.cvtColor(lab_colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 255).astype("uint8")

# Generate output filename
output_image_path = f"colorized_{os.path.basename(image_path)}"
cv2.imwrite(output_image_path, colorized)

# Show the results
cv2.imshow("Original Image", image)
cv2.imshow("Colorized Image", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Colorized image saved at: {output_image_path}")
