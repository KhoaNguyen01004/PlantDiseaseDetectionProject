import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import logging
import subprocess
import tensorflow as tf
import numpy as np

# Fix for module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess import load_config, build_dataloaders, export_to_tflite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_data(cfg_path='configs/config.yaml'):
    """Load config, dataloaders, model with best weights. Safety checks."""
    if not os.path.exists(cfg_path):
        logger.error(f'Config missing: {cfg_path}')
        sys.exit(1)
    
    cfg = load_config(cfg_path)
    dataloaders, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)
    logger.info(f'Loaded data with {num_classes} classes.')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modern Weights enum - Must match train.py architecture (EfficientNet-B2)
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    classifier = model.classifier
    in_features = classifier[1].in_features
    model.classifier = nn.Sequential(classifier[0], nn.Linear(in_features, num_classes))
    
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        logger.error(f'Model missing: {model_path}. Run train.py first.')
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f'Model loaded from {model_path} to {device}')
    
    model.to(device).eval()
    return model, dataloaders, label_map, device, cfg

def run_evaluation(model, dataloaders, device, label_map):
    """Run test evaluation with metrics."""
    logger.info('Running test evaluation...')
    y_true, y_pred = [], []
    rev_label_map = {v: k for k, v in label_map.items()}
    target_names = [rev_label_map[i] for i in range(len(label_map))]
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Test Eval'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred) * 100
    logger.info('Test Accuracy: %.2f%%', acc)
    print('\nClassification Report:\n', classification_report(y_true, y_pred, target_names=target_names, digits=4))
    return acc

def export_onnx(model, image_size=224):
    """Export model to ONNX."""
    onnx_file = 'plant_model.onnx'
    logger.info(f'Exporting to ONNX: {onnx_file}')
    model_cpu = model.cpu()
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(
        model_cpu, 
        (dummy_input,), 
        onnx_file,
        opset_version=15, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        export_params=True
    )
    logger.info('ONNX export complete.')
    return onnx_file

def convert_to_tflite_float32(onnx_file):
    """Convert ONNX to float32 TFLite via onnx2tf."""
    try:
        logger.info('Converting to float32 TFLite via onnx2tf...')
        output_dir = 'plant_model_tflite_float32'
        
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            
        cmd = ["onnx2tf", "-i", onnx_file, "-o", output_dir, "--non_verbose"]
        logger.info('Running: %s', ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info('✅ Float32 TFLite conversion successful.')
            return os.path.join(output_dir, 'plant_model.tflite')
        else:
            logger.error('onnx2tf failed (code %d): %s', result.returncode, result.stderr)
            return None
    except Exception as e:
        logger.error(f'Float32 conversion failed: {e}')
        return None

def verify_tflite(tflite_path):
    """Verify TFLite model integrity and size."""
    if tflite_path is None or not os.path.exists(tflite_path):
        logger.warning(f'TFLite verification skipped: {tflite_path}')
        return
    
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        logger.info('✅ TFLite verified.')
        size_mb = os.path.getsize(tflite_path) / (1024**2)
        logger.info(f'Model size: {tflite_path} ({size_mb:.1f} MB)')
    except Exception as e:
        logger.warning(f'Verification failed: {e}')

def calibration_dataset(dataloaders, cfg, num_calib=100):
    """Generator for PTQ calibration using 100 test images, uint8 input."""
    test_loader = dataloaders['test']
    calib_count = 0
    image_size = cfg['image_size']
    
    for inputs, _ in test_loader:
        for img in inputs:
            # Denormalize to uint8 [0,255]
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img.permute(1,2,0).numpy()
            img_np = img_np * std + mean
            img_np = (img_np * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (2,0,1))  # HWC to CHW
            img_np = np.expand_dims(img_np, 0)
            yield [img_np]
            
            calib_count += 1
            if calib_count >= num_calib:
                return

def convert_to_tflite_int8(onnx_file, dataloaders, cfg):
    """PTQ INT8 with uint8 input using 100 test images."""
    try:
        logger.info('Converting to INT8 TFLite with PTQ...')
        
        # ONNX -> TF
        import onnx_tf
        model_tf = onnx_tf.backend.prepare(onnx_tf.backend.load_model(onnx_file))
        tf_model_path = 'plant_model_tf'
        model_tf.export_graph(tf_model_path)
        
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: calibration_dataset(dataloaders, cfg)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        
        tflite_int8 = converter.convert()
        tflite_path = 'plant_model_tflite_int8/plant_model_int8.tflite'
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_int8)
        
        logger.info('✅ INT8 TFLite PTQ successful.')
        verify_tflite(tflite_path)
        return tflite_path
    except Exception as e:
        logger.error(f'INT8 PTQ failed: {e}')
        return None

def main():
    try:
        model, dataloaders, label_map, device, cfg = load_model_and_data()
        run_evaluation(model, dataloaders, device, label_map)
        
        onnx_file = export_onnx(model, cfg['image_size'])
        tflite_float = convert_to_tflite_float32(onnx_file)
        verify_tflite(tflite_float)
        
        tflite_int8 = convert_to_tflite_int8(onnx_file, dataloaders, cfg)
        verify_tflite(tflite_int8)
        
        logger.info('Pipeline complete: float32 + INT8 ready.')
    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()

