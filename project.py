import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return cls(**config)

def preprocess_image(image_path):
    """
    Enhanced preprocessing with standardization
    """
    try:
        # Load image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
        # Convert to RGB if needed
        if img_array.shape[-1] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Standardize the image
        mean = np.mean(img_array)
        std = np.std(img_array)
        img_array = (img_array - mean) / (std + 1e-7)
        
        # Expand dimensions for batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def analyze_image_density(image_path):
    """
    Enhanced density analysis with multiple features
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        # Resize for consistent analysis
        img = cv2.resize(img, (224, 224))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Calculate various features
        
        # 1. Edge density using Canny
        edges = cv2.Canny(enhanced, 100, 200)
        edge_density = np.sum(edges == 255) / edges.size
        
        # 2. Texture analysis using LBP
        radius = 1
        n_points = 8 * radius
        lbp = cv2.imread(image_path, 0)  # Read as grayscale
        lbp = cv2.resize(lbp, (224, 224))
        
        # Calculate pixel-wise variance
        variance = np.var(lbp)
        normalized_variance = variance / (255 * 255)  # Normalize to [0,1]
        
        # 3. Vessel density using threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_density = np.sum(binary == 255) / binary.size
        
        # Combine features
        density_score = (edge_density * 0.4 + 
                        normalized_variance * 0.3 + 
                        vessel_density * 0.3)
        
        return {
            'overall_density': density_score,
            'edge_density': edge_density,
            'texture_variance': normalized_variance,
            'vessel_density': vessel_density,
            'binary_image': binary
        }
    
    except Exception as e:
        print(f"Error in density analysis: {str(e)}")
        return None

def classify_disease(model, image_path):
    """
    Improved classification with multiple thresholds
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get model prediction
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None

        predictions = model.predict(img_array, verbose=0)
        print("\nRaw prediction values:", predictions[0])
        
        predicted_class = np.argmax(predictions[0])
        class_confidence = predictions[0][predicted_class]
        
        # Get density analysis
        density_results = analyze_image_density(image_path)
        if density_results is None:
            return None

        # Class mapping
        class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
        original_class = class_names[predicted_class]
        
        # Enhanced decision logic
        no_dr_confidence = predictions[0][0]  # Confidence for No_DR class
        
        # Define thresholds
        CONFIDENCE_THRESHOLD = 0.4
        DENSITY_THRESHOLD = 0.15
        EDGE_DENSITY_THRESHOLD = 0.1
        
        # Decision logic for No Disease
        is_no_disease = (
            (no_dr_confidence > CONFIDENCE_THRESHOLD) or
            (density_results['overall_density'] < DENSITY_THRESHOLD and
             density_results['edge_density'] < EDGE_DENSITY_THRESHOLD and
             predicted_class <= 1)  # No_DR or Mild
        )
        
        result = {
            'diagnosis': 'No Disease' if is_no_disease else 'Disease',
            'confidence_score': float(class_confidence),
            'density_scores': {
                'overall': float(density_results['overall_density']),
                'edge': float(density_results['edge_density']),
                'texture': float(density_results['texture_variance']),
                'vessel': float(density_results['vessel_density'])
            },
            'detailed_class': original_class,
            'class_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, predictions[0])
            },
            'binary_image': density_results['binary_image']
        }
        
        return result

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None

def display_results(image_path, results):
    """
    Enhanced visualization with more detailed information
    """
    try:
        # Read original image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (600, 400))
        
        # Create result visualization
        display_img = np.zeros((900, 1200, 3), dtype=np.uint8)
        
        # Place original image
        display_img[50:450, 50:650] = img
        
        # Place binary image
        if results['binary_image'] is not None:
            binary_display = cv2.cvtColor(results['binary_image'], cv2.COLOR_GRAY2BGR)
            binary_display = cv2.resize(binary_display, (600, 400))
            display_img[50:450, 650:1250] = binary_display
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) if results['diagnosis'] == 'No Disease' else (0, 0, 255)
        
        # Main results
        cv2.putText(display_img, f"Diagnosis: {results['diagnosis']}", 
                    (50, 500), font, 1, color, 2)
        cv2.putText(display_img, f"Detailed Class: {results['detailed_class']}", 
                    (50, 550), font, 1, (255, 255, 255), 2)
        
        # Density scores
        y_pos = 600
        cv2.putText(display_img, "Density Scores:", 
                    (50, y_pos), font, 1, (255, 255, 255), 2)
        for name, score in results['density_scores'].items():
            y_pos += 40
            cv2.putText(display_img, f"{name}: {score:.3f}", 
                        (50, y_pos), font, 0.7, (255, 255, 255), 2)
        
        # Class probabilities
        y_pos = 600
        cv2.putText(display_img, "Class Probabilities:", 
                    (650, y_pos), font, 1, (255, 255, 255), 2)
        for class_name, prob in results['class_probabilities'].items():
            y_pos += 40
            cv2.putText(display_img, f"{class_name}: {prob:.3f}", 
                        (650, y_pos), font, 0.7, (255, 255, 255), 2)
        
        # Display the results
        cv2.imshow('Analysis Results', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error displaying results: {str(e)}")

def main():
    model_path = r'C:\Users\aravi\Downloads\converted_keras (1)\keras_model.h5'
    
    try:
        print("Loading model...")
        model = load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        print("\nModel Summary:")
        model.summary()
        
        while True:
            print("\nEnter the path to the retinal image (or 'q' to quit):")
            image_path = input().strip()
            
            if image_path.lower() == 'q':
                break
                
            if not os.path.exists(image_path):
                print("Image file not found. Please check the path and try again.")
                continue
            
            print("\nProcessing image...")
            results = classify_disease(model, image_path)
            
            if results:
                print("\nAnalysis Results:")
                print(f"Diagnosis: {results['diagnosis']}")
                print(f"Detailed Class: {results['detailed_class']}")
                print("\nDensity Scores:")
                for name, score in results['density_scores'].items():
                    print(f"{name}: {score:.3f}")
                print("\nClass Probabilities:")
                for class_name, prob in results['class_probabilities'].items():
                    print(f"{class_name}: {prob:.3f}")
                
                display_results(image_path, results)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()