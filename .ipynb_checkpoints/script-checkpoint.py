import face_recognition
import cv2
import numpy as np
import os
from deepface import DeepFace

def load_and_encode_faces(image_paths):
    """Load images, detect faces, and return encodings."""
    encodings = {}
    for img_path in image_paths:
        try:
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            
            if encoding:
                encodings[img_path] = encoding[0]  # Store first detected face encoding
            else:
                print(f"No face found in {img_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return encodings

def find_non_matching_faces(encodings):
    """Compare all faces and find non-matching ones."""
    reference_encoding = None
    mismatched = []

    for img_path, encoding in encodings.items():
        if reference_encoding is None:
            reference_encoding = encoding  # Use first image as reference
            continue
        
        match = face_recognition.compare_faces([reference_encoding], encoding, tolerance=0.5)
        if not match[0]:
            mismatched.append(img_path)
    
    return mismatched

def detect_gender(image_paths):
    """Predict gender for each image using DeepFace."""
    gender_results = {}
    
    for img_path in image_paths:
        try:
            result = DeepFace.analyze(img_path, actions=["gender"], enforce_detection=False)
            gender = result[0]['dominant_gender']
            gender_results[img_path] = gender
        except Exception as e:
            print(f"Gender detection failed for {img_path}: {str(e)}")
    
    return gender_results

def main(image_folder):
    """Main function to process images for face matching and gender detection."""
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
    if len(image_paths) > 10:
        image_paths = image_paths[:10]  # Limit to 10 images

    print(f"Processing {len(image_paths)} images...")

    # Step 1: Encode faces
    encodings = load_and_encode_faces(image_paths)
    
    if not encodings:
        print("No valid faces found. Exiting...")
        return

    # Step 2: Find mismatched images
    mismatched_images = find_non_matching_faces(encodings)
    
    # Step 3: Detect gender
    gender_results = detect_gender(image_paths)

    # Print results
    print("\n✅ Face Matching Results:")
    if mismatched_images:
        print("Non-matching images:", mismatched_images)
    else:
        print("All faces match!")

    print("\nGender Prediction:")
    for img_path, gender in gender_results.items():
        print(f"{os.path.basename(img_path)} → {gender}")

if __name__ == "__main__":
    image_folder = "images"  # Change this to your folder path
    main(image_folder)
