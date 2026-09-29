import torch
from PIL import Image
from torchvision import transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src\vit_13_waste_classifier.pth"

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

detector = YOLO('yolov8n.pt') 

CLASS_LABELS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                 'metal', 'non_waste','paper', 'plastic', 'shoes', 'trash', 'white-glass']

def process_user_image_viT(image_path):
   
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # 3. DETECT: Find all distinct items in the image
    # We use agnostic_nms=True so overlapping bounding boxes don't double-detect the same trash
    results = detector(image, agnostic_nms=True)[0]
    
    print(f"--- Found {len(results.boxes)} items in the photo. Starting classification... ---")
    
    # 4. INLOOP: Processing each detected item one by one
    if(len(results.boxes)>0):
        for idx, box in enumerate(results.boxes):
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add a tiny 5-pixel boundary padding so MobileNet can see the edges clearly
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)
            
            #extracting individual images
            cropped_waste = image[y1:y2, x1:x2]
            
            # 6. PREPROCESSING FOR MOBILENET: Convert BGR to RGB and resize to MobileNet's expected input shape
            cropped_rgb = cv2.cvtColor(cropped_waste, cv2.COLOR_BGR2RGB)
            resized_item = cv2.resize(cropped_rgb, (224, 224))
            
            input_tensor = np.expand_dims(resized_item.astype(np.float32), axis=0)

            # 7. PREDICTING croped images
            predictions = predict_waste(resized_item)
            predicted_class =predictions[0]
            confidence = predictions[1]
            
      
            
            print(f"Item {idx + 1}: Classified as [{predicted_class}] with {confidence*100:.1f}% confidence.")
            
            # OPTIONAL: Draw the predicted label on the original image for the user to see
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Calculate text size to ensure exact positioning
            (text_width, text_height), baseline = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            
            #showing predicted class text in images
            if y1 - 15 < 0:
              text_y = y1 + text_height + 10 
            else:
              text_y = y1 - 15 

            cv2.putText(image, predicted_class, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


        cv2.imwrite('user_output_classified.jpg', image)
        return  image

    else:
      return  predict_waste(image_path)









CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
     'green-glass', 'metal', 'non_waste', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]
NUM_CLASSES = len(CLASS_NAMES)


model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("src/vit_13_waste_classifier.pth",map_location=torch.device('cpu')))
model = model.to(device)
model.eval()  # Crucial: Sets layers like Dropout to evaluation mode

# 3. Define the Validation Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 4. Prediction Function
def predict_waste(image_path):

    if isinstance(image_path, str):
      image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, np.ndarray):
      image = Image.fromarray(image_path).convert('RGB')
    else:
      image = image_path.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Apply Softmax to convert raw logits into percentages
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Find the index with the highest probability
        confidence, class_idx = torch.max(probabilities, dim=0)
        
    predicted_class = CLASS_NAMES[class_idx.item()]
    confidence_score = confidence.item() * 100
    
    print(f"\n----------Prediction using visison transformer:-----------")
    print(f"Result: {predicted_class.upper()}")
    print(f"Confidence: {confidence_score:.2f}%")
    
    return predicted_class, confidence_score
