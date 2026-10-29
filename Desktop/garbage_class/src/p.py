import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from src.vitP import predict_waste
from src.predict import predict_waste

detector = None
mobilenet_classifier = None

CLASS_LABELS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                 'metal', 'non_waste','paper', 'plastic', 'shoes', 'trash', 'white-glass']

# defining yolo function, it will load only when user uploads image 
def get_yolo_detector():
    global detector
    if detector is None:
        print("--- Initializing YOLO detector ---")
        detector = YOLO('yolov8n.pt')
    return detector

# same way defining model, initialize at first prediction
def get_mobilenet_classifier():
    global mobilenet_classifier
    if mobilenet_classifier is None:
        print("--- Initializing MobileNet Classifier ---")
        mobilenet_classifier = tf.keras.models.load_model("src/mobileWaste_classifier.keras")
    return mobilenet_classifier


def process_user_image(image_path):
    
    yolo_model = get_yolo_detector()
    classifier_model = get_mobilenet_classifier()   
  
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    #  Find all distinct items in the image using the variable instance
    results = yolo_model(image, agnostic_nms=True)[0]
    
    print(f"--- Found {len(results.boxes)} items in the photo. Starting classification... ---")
    
    # Process each detected item one by one
    if(len(results.boxes) > 0):
        for idx, box in enumerate(results.boxes):
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add a tiny 5-pixel boundary padding so MobileNet can see the edges clearly
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)
            
            #  Crop out the individual piece of waste
            cropped_waste = image[y1:y2, x1:x2]
            
            #  PREPROCESS FOR MOBILENET: Convert BGR to RGB and resize to MobileNet's expected input shape
            cropped_rgb = cv2.cvtColor(cropped_waste, cv2.COLOR_BGR2RGB)
            resized_item = cv2.resize(cropped_rgb, (224, 224))
            
            input_tensor = np.expand_dims(resized_item.astype(np.float32), axis=0)

            # predict the cropped single image 
            predictions = classifier_model.predict(input_tensor)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            label = CLASS_LABELS[predicted_class_idx]
            
            print(f"Item {idx + 1}: Classified as [{label}] with {confidence*100:.1f}% confidence.")
            
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            

            if y1 - 15 < 0:
              text_y = y1 + text_height + 10 
            else:
              text_y = y1 - 15 

            cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imwrite('user_output_classified.jpg', image)
        return image

    else:
      return predict_waste(image_path)
