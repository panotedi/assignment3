import streamlit as st
from transformers import pipeline
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
import glob
import os

resnet_model = ResNet50(weights='imagenet')

st.title("CS634 - Assignment 3")

user_image_input = st.file_uploader("Upload Images", type=["jpg"])


path='lfw2/V*'
photos=[]
for fold in glob.glob(path, recursive=True):
    for subdir, dirs, files in os.walk(fold):
        for file in files:
            #st.write(file)
            photos.append(os.path.join(subdir, file))
            
photos.insert(0,"")
celebrity_photo = st.selectbox("Select Photo",photos)

            
            
def extract_features(photos, resnet_model):
    features = {}
    for photo in photos:
        if(photo!=""):
            img = image.load_img(photo, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            features_vector = resnet_model.predict(x)
            features_vector = features_vector.flatten()
            features[photo] = features_vector

    return features
            

            
            
if(len(celebrity_photo) != 0):
    #st.image(user_image_input, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    user_input_image = None
    st.write(celebrity_photo)
    #st.write(user_image_input.read())
    size=len(photos)
    #st.write(size)
    st.write("Query Image: ")
    st.image(celebrity_photo)
    features = extract_features(photos, resnet_model)
    
    features_array = np.array(list(features.values()))

    nn_model = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nn_model.fit(features_array)

    query_image_path = photos[size-1]
    query_image_feature = features[query_image_path].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_image_feature)
    
    
    st.write("Similar Images:")
    for i in range(1,11):
        similar_image_path = photos[indices[0][i]]
        similar_image_distance = distances[0][i]
        st.write("Similar Image #{}: Distance: {}".format(i, similar_image_distance))
        st.image(similar_image_path)
        

if(user_image_input != None):
    celebrity_photo = []
    #st.image(user_image_input, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    im = Image.open(user_image_input)
    im=im.resize((224,224))
    im.save("input_image.jpg", "JPEG")
    photos.append("input_image.jpg")
    #st.write(user_image_input.read())
    size=len(photos)
    #st.write(size)
    st.write("Query Image: ")
    st.image(photos[size-1])
    features = extract_features(photos, resnet_model)
    
    features_array = np.array(list(features.values()))

    nn_model = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nn_model.fit(features_array)

    query_image_path = photos[size-1]
    query_image_feature = features[query_image_path].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_image_feature)
    
    
    st.write("Similar Images:")
    for i in range(1,11):
        similar_image_path = photos[indices[0][i]]
        similar_image_distance = distances[0][i]
        st.write("Similar Image #{}: Distance: {}".format(i, similar_image_distance))
        st.write(similar_image_path)
        st.image(similar_image_path)
#else:
#    size=len(photos)
#    st.write(size)
#    st.image(photos[size-1])
#    features = extract_features(photos, resnet_model)
            
            
            


