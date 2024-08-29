import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


# set title
st.title('Teath classification')

# set header
st.header('Please upload a image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model("C:\\Users\\Hp\\Downloads\\Teeth Classification2.h5")

# load class names
with open("C:\\Users\\Hp\\Downloads\\Teeth Classification2.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))