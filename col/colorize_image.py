import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Function to colorize the black & white image
def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Update paths (Ensure these are correct)
    prototxt = r"C:\Users\Siva\Desktop\Colorizer\models\models_colorization_deploy_v2.prototxt"
    model = r"C:\Users\Siva\Desktop\Colorizer\models\colorization_release_v2.caffemodel"
    points = r"C:\Users\Siva\Desktop\Colorizer\models\pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert image to LAB color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

#######################################################################################

st.title("Colorize Your Black & White Image")
st.write("Upload a B&W image to get a colorized version!")

file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file is None:
    st.warning("No image uploaded yet. Please upload an image to proceed.")
else:
    try:
        image = Image.open(file)
        img = np.array(image)

        # Convert RGB to BGR (since OpenCV uses BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        st.subheader("Colorized Image")
        color = colorizer(img)

        st.image(color, use_column_width=True)

        st.success("Colorization completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
