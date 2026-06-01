#!/usr/bin/env python3
"""Real-world inference for Plant Disease model - Samsung A17 5G optimized.

Supports both float32 and INT8 TFLite models with dynamic input type detection.
"""
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm
import logging
from scipy.special import softmax
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}

    return {}

_CONFIG = load_config()
IMAGE_SIZE = _CONFIG.get("image", {}).get("size", 260)
MEAN = np.array(_CONFIG.get("image", {}).get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(_CONFIG.get("image", {}).get("std", [0.229, 0.224, 0.225]), dtype=np.float32)


def preprocess_image(img_path_or_frame, image_size=IMAGE_SIZE, input_dtype=np.float32):
    """Preprocess image for TFLite inference.
    
    Args:
        img_path_or_frame: Path to image or cv2 frame
        image_size: Target size for resizing
        input_dtype: Expected input dtype from TFLite model (float32 or uint8)
    
    Returns:
        Preprocessed image tensor
    """
    if isinstance(img_path_or_frame, str):
        img = cv2.imread(img_path_or_frame)
        if img is None:
            raise ValueError(f'Cannot load image: {img_path_or_frame}')
    else:
        img = img_path_or_frame.copy()
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (image_size, image_size))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img = (img - MEAN) / STD
    
    # Convert to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Cast to model's expected input type
    if input_dtype == np.uint8:
        # For INT8 models: scale to [0, 255] and cast to uint8
        img = (img * 255).astype(np.uint8)
    else:
        # For float32 models: keep as float32 (already normalized)
        img = img.astype(np.float32)
    
    return img


def load_labels(labels_path='labels.json'):
    """Load class labels from JSON file."""
    with open(labels_path) as f:
        return json.load(f)


def create_delegates(model_path):
    """Create TFLite delegates for hardware acceleration.
    
    A17 5G: GPU Delegate (Mali-G68) primary, XNNPACK CPU fallback.
    """
    delegates = []
    
    # GPU Delegate
    try:
        gpu_delegate = tf.lite.experimental.load_delegate('libgpu_delegate.so')  # Android
        delegates.append(gpu_delegate)
        logger.info('GPU Delegate (Mali-G68) enabled')
    except Exception as e:
        logger.warning('GPU Delegate unavailable, using CPU: %s', e)
    
    # XNNPACK CPU fallback
    try:
        delegates.append(tf.lite.experimental.load_delegate('libxnnpack.so'))
        logger.info('XNNPACK CPU delegate enabled')
    except Exception as e:
        logger.warning('XNNPACK delegate unavailable: %s', e)
    
    return delegates


def detect_input_details(input_details):
    """Detect model input type and parameters.
    
    Returns:
        tuple: (input_dtype, input_scale, input_zero_point)
    """
    input_details = input_details[0]
    input_dtype = input_details['dtype']
    input_scale = input_details.get('quantization', (1.0, 0))[0]
    input_zero_point = input_details.get('quantization', (1.0, 0))[1]
    
    logger.info(f'Model input dtype: {input_dtype}')
    logger.info(f'Model quantization: scale={input_scale}, zero_point={input_zero_point}')
    
    return input_dtype, input_scale, input_zero_point


def run_inference(model_path, img_path=None, webcam=False, labels=None):
    """Memory-efficient inference with dynamic input type detection."""
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=create_delegates(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Detect model input type dynamically
    input_dtype, input_scale, input_zero_point = detect_input_details(input_details)
    
    input_shape = input_details[0]['shape']
    logger.info(f'Input shape: {input_shape}')
    
    if img_path:
        # Preprocess with correct dtype
        input_data = preprocess_image(img_path, input_dtype=input_dtype)
        
        # For quantized models, apply quantization parameters
        if input_dtype == np.uint8 and (input_scale != 1.0 or input_zero_point != 0):
            input_data = (input_data / input_scale) + input_zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Apply softmax for proper probability distribution
        probabilities = softmax(output_data)
        
        # Get prediction and confidence
        pred_id = np.argmax(probabilities)
        conf = probabilities[pred_id]
        pred_class = labels[str(pred_id)] if isinstance(labels, dict) else labels[pred_id]
        
        logger.info(f'Prediction: {pred_class} (confidence: {conf*100:.1f}%)')
        logger.info(f'Top-3 predictions:')
        top3_indices = np.argsort(probabilities)[::-1][:3]
        for idx in top3_indices:
            class_name = labels[str(idx)] if isinstance(labels, dict) else labels[idx]
            logger.info(f'  {class_name}: {probabilities[idx]*100:.2f}%')
        
        return pred_class, conf
    
    elif webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            input_data = preprocess_image(frame, input_dtype=input_dtype)
            
            # For quantized models, apply quantization parameters
            if input_dtype == np.uint8 and (input_scale != 1.0 or input_zero_point != 0):
                input_data = (input_data / input_scale) + input_zero_point
                input_data = np.clip(input_data, 0, 255).astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Apply softmax for proper probability distribution
            probabilities = softmax(output_data)
            
            pred_id = np.argmax(probabilities)
            conf = probabilities[pred_id]
            pred_class = labels[str(pred_id)] if isinstance(labels, dict) else labels[pred_id]
            
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
    parser.add_argument('--labels', default='labels.json', help='Path to labels file')
    args = parser.parse_args()
    
    if not args.image and not args.webcam:
        parser.error('Provide --image or --webcam')
    
    if not os.path.exists(args.model):
        logger.error(f'Model not found: {args.model}')
        return
    
    labels = load_labels(args.labels)
    logger.info(f'Loaded {len(labels)} classes from {args.labels}')
    
    run_inference(args.model, args.image, args.webcam, labels)


if __name__ == '__main__':
    main()