#!/usr/bin/env python3
"Real-world inference for Plant Disease model - Samsung A17 5G optimized."
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMAGE_SIZE = 224

def preprocess_image(img_path_or_frame, image_size=IMAGE_SIZE):
    "cv2 preprocess matching training: resize, ImageNet norm -> uint8 for INT8."
    if isinstance(img_path_or_frame, str):
        img = cv2.imread(img_path_or_frame)
        if img is None:
            raise ValueError(f'Cannot load image: {img_path_or_frame}')
    else:
        img = img_path_or_frame.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD  # Normalize
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0).astype(np.uint8)  # INT8 input, [1,3,224,224] uint8
    return img

def load_labels(labels_path='labels.json'):
    with open(labels_path) as f:
        return json.load(f)

def create_delegates(model_path):
    "A17 5G: GPU Delegate (Mali-G68) primary, XNNPACK CPU fallback."
    delegates = []
    
    # GPU Delegate
    try:
        gpu_delegate = tf.lite.experimental.load_delegate('libgpu_delegate.so')  # Android
        delegates.append(gpu_delegate)
        logger.info('GPU Delegate (Mali-G68) enabled')
    except:
        logger.warning('GPU Delegate unavailable, using CPU')
    
    # XNNPACK CPU fallback
    delegates.append(tf.lite.experimental.load_delegate('libxnnpack.so'))
    logger.info('XNNPACK CPU delegate enabled')
    
    return delegates

def run_inference(model_path, img_path=None, webcam=False, labels=None):
    "Memory-efficient inference."
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=create_delegates(model_path))
    interpreter.allocate_tensors()  # Single allocation
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    logger.info(f'Input shape: {input_shape}')
    
    if img_path:
        input_data = preprocess_image(img_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        conf = np.max(output_data)
        pred_id = np.argmax(output_data)
        pred_class = labels[pred_id]
        logger.info(f'Prediction: {pred_class} ({conf*100:.1f}%)')
        return pred_class, conf
    
    elif webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            input_data = preprocess_image(frame)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            conf = np.max(output_data)
            pred_id = np.argmax(output_data)
            pred_class = labels[pred_id]
            
            cv2.putText(frame, f'{pred_class}: {conf*100:.1f}%', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Plant Disease Detection - A17 5G', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Inference (A17 optimized)')
    parser.add_argument('--model', default='plant_model_tflite_int8/plant_model_int8.tflite', 
                       help='TFLite model (float32 or int8)')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--webcam', action='store_true', help='Use camera')
    args = parser.parse_args()
    
    if not args.image and not args.webcam:
        parser.error('Provide --image or --webcam')
    
    if not os.path.exists(args.model):
        logger.error(f'Model not found: {args.model}')
        return
    
    labels = load_labels()
    logger.info(f'Loaded {len(labels)} classes from labels.json')
    
    run_inference(args.model, args.image, args.webcam, labels)

if __name__ == '__main__':
    main()

