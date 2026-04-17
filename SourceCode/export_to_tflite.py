#!/usr/bin/env python3
"""
Export PyTorch model to TFLite INT8 for inference.py compatibility.
Compatible with EfficientNet_B0 from train.py.
"""
import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import argparse
from torchvision import models
from tqdm import tqdm
import os
import shutil
import json

# Config consistent with training
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def load_labels(labels_path='labels.json'):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return len(labels)

def load_pytorch_model(model_path='models/best_model.pth'):
    """Load EfficientNet_B0 model trained with train.py."""
    num_classes = load_labels()
    
    model = models.efficientnet_b0(pretrained=False)
    classifier = model.classifier
    in_features = classifier[1].in_features.item() if hasattr(classifier[1].in_features, 'item') else classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        classifier[0], 
        torch.nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def export_to_onnx(model, onnx_path='plant_model.onnx'):
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f'Exported to {onnx_path}')

def create_saved_model(onnx_path):
    """Convert ONNX to SavedModel."""
    if os.path.exists('temp_saved_model'):
        shutil.rmtree('temp_saved_model')
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    tf_rep = prepare(onnx_model)
    saved_model_dir = 'temp_saved_model'
    tf_rep.export_graph(saved_model_dir)
    print(f'SavedModel created at {saved_model_dir}')
    return saved_model_dir

def quantize_to_int8(saved_model_dir, tflite_int8_path='plant_model_int8.tflite', calib_images_dir='data/val'):
    """Post-training INT8 quantization with TF representative dataset."""
    
    def representative_data_gen():
        calib_files = []
        for root, _, files in os.walk(calib_images_dir):
            calib_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        calib_files = calib_files[:100]  # Max 100 calibration samples
        
        print(f'Using {len(calib_files)} calibration images')
        for img_path in calib_files:
            try:
                # Preprocess matching inference.py exactly
                img_data = tf.io.read_file(img_path)
                img_data = tf.io.decode_image(img_data, channels=3)
                img_data = tf.image.resize(img_data, [IMAGE_SIZE, IMAGE_SIZE])
                img_data = tf.cast(img_data, tf.float32) / 255.0
                img_data = (img_data - MEAN) / STD
                img_data = tf.transpose(img_data, perm=[2, 0, 1])  # HWC -> CHW
                img_data = tf.expand_dims(img_data, 0)  # batch dim
                img_data = tf.cast(img_data, tf.uint8)
                yield [img_data.numpy()]
            except Exception as e:
                print(f'Skip {img_path}: {e}')
                continue
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  
    converter.inference_output_type = tf.float32
    converter.allow_custom_ops = True
    
    tflite_quant_model = converter.convert()
    
    # Save INT8 model
    with open(tflite_int8_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f'✅ INT8 TFLite model saved: {tflite_int8_path}')
    
    # Verify model
    interpreter = tf.lite.Interpreter(model_path=tflite_int8_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print('Input details:', input_details)
    print('Output details:', output_details)
    
    # Test with dummy input
    dummy_input = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f'Test inference shape: {output.shape}, min: {output.min():.3f}, max: {output.max():.3f}')
    print('Export pipeline complete! Model ready for inference.py')

def main():
    parser = argparse.ArgumentParser(description='PyTorch -> ONNX -> TFLite INT8')
    parser.add_argument('--model', default='models/best_model.pth', help='PyTorch checkpoint')
    parser.add_argument('--data-dir', default='data/val', help='Calibration/validation images dir')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep temp files')
    args = parser.parse_args()
    
    print('1. Loading PyTorch model...')
    model = load_pytorch_model(args.model)
    
    print('2. Exporting to ONNX...')
    export_to_onnx(model)
    
    print('3. Converting ONNX to SavedModel...')
    saved_model_dir = create_saved_model('plant_model.onnx')
    
    print('4. Quantizing to TFLite INT8...')
    quantize_to_int8(saved_model_dir, calib_images_dir=args.data_dir)
    
    # Cleanup
    if not args.no_cleanup:
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        os.remove('plant_model.onnx')
        print('Cleaned up temp files')
    
    print('\n🎉 All done! Use python src/inference.py --model plant_model_int8.tflite')

if __name__ == '__main__':
    main()
