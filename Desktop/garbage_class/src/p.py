import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from src.vitP import predict_waste
from src.predict import predict_waste

detector = YOLO('yolov8n.pt') 
mobilenet_classifier = tf.keras.models.load_model("src\mobileWaste_classifier.keras")


CLASS_LABELS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                 'metal', 'non_waste','paper', 'plastic', 'shoes', 'trash', 'white-glass']

def process_user_image(image_path):
   
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # 3. DETECT: Find all distinct items in the image
    # We use agnostic_nms=True so overlapping bounding boxes don't double-detect the same trash
    results = detector(image, agnostic_nms=True)[0]
    
    print(f"--- Found {len(results.boxes)} items in the photo. Starting classification... ---")
    
    # 4. LOOP: Process each detected item one by one
    if(len(results.boxes)>0):
        for idx, box in enumerate(results.boxes):
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add a tiny 5-pixel boundary padding so MobileNet can see the edges clearly
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)
            
            # 5. EXTRACT: Crop out the individual piece of waste
            cropped_waste = image[y1:y2, x1:x2]
            
            # 6. PREPROCESS FOR MOBILENET: Convert BGR to RGB and resize to MobileNet's expected input shape
            # (Most MobileNet models expect 224x224 pixels)
            cropped_rgb = cv2.cvtColor(cropped_waste, cv2.COLOR_BGR2RGB)
            resized_item = cv2.resize(cropped_rgb, (224, 224))
            
            input_tensor = np.expand_dims(resized_item.astype(np.float32), axis=0)

            # 7. PREDICT: Pass the single clean crop to your MobileNet model
            predictions = mobilenet_classifier.predict(input_tensor)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            label = CLASS_LABELS[predicted_class_idx]
            
            print(f"Item {idx + 1}: Classified as [{label}] with {confidence*100:.1f}% confidence.")
            
            # OPTIONAL: Draw the predicted label on the original image for the user to see
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Calculate text size to ensure exact positioning
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            
            #showing predicted class text in images
            if y1 - 15 < 0:
              text_y = y1 + text_height + 10 
            else:
              text_y = y1 - 15 

            cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


        cv2.imwrite('user_output_classified.jpg', image)
        return  image

    else:
      return  predict_waste(image_path)