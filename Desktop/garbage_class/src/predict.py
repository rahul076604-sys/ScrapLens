import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("src\mobileWaste_classifier.keras")

class_names = np.load("src\class_names.npy", allow_pickle=True)


def predict_waste(filepath):
    image_path =filepath
    IMG_SIZE = (224, 224)

    img1 = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array1 = tf.keras.utils.img_to_array(img1)
    img_array1 = np.expand_dims(img_array1, axis=0)


    # Predict
    predictions1 = model.predict(img_array1)


    predicted_index1 = np.argmax(predictions1[0])
    predicted_class1 = class_names[predicted_index1]
    confidence1 = np.max(predictions1[0])

    print("Prediction percentage for each class: ",predictions1)
    print("Prediction:", predicted_class1)
    print("Confidence:", round(confidence1 * 100, 2), "%")
    return {
        "class":predicted_class1,"confidence":round(confidence1 * 100, 2)}
