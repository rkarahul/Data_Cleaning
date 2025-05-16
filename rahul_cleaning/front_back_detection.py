import numpy as np
import cv2
import random
from openvino.runtime import Core, get_version, PartialShape
import time
from logfile import logger

class FrontBackDetection:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = Core()
        available_devices = self.core.available_devices
        print(f"OpenVINO version: {get_version()}")
        print(f"Available devices: {available_devices}")
        
        if device.startswith("GPU") and not any(d.startswith("GPU") for d in available_devices):
            print("WARNING: GPU device requested but not available. Falling back to CPU.")
            device = "CPU"
        
        self.model = self.core.read_model(model_path)
        
        self.target_height = 64
        self.target_width = 64
        self.channels = 3
        
        if self.model.inputs[0].partial_shape.is_dynamic:
            self.model.reshape({self.model.inputs[0]: PartialShape([1, self.channels, self.target_height, self.target_width])})
        
        if device.startswith("GPU"):
            gpu_config = {
                "CACHE_DIR": "./gpu_cache",
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "FP16"
            }
            self.compiled_model = self.core.compile_model(self.model, device, gpu_config)
        else:
            self.compiled_model = self.core.compile_model(self.model, device)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")
        
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3 channels, got shape {image.shape}")
        
        resized = cv2.resize(image, (self.target_width, self.target_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        
        return batch

    def run_inference(self, image: np.ndarray) -> str:
        try:
            preprocessed = self.preprocess_image(image)
            result = self.compiled_model({self.input_layer.any_name: preprocessed})
            predictions = result[self.output_layer][0]
            
            class_index = np.argmax(predictions)  # Get the class with the highest probability
            
            class_labels = ["back", "front", "unknown"]  # Updated class labels
            return class_labels[class_index]
        
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return "unknown"
    
    def find_vehicle_orientation(self, frames):
        result_list = []
        num = len(frames)
        
        if num == 0:
            return "unknown"
        
        for frame in frames:
            output = self.run_inference(frame)
            if output is not None:
                result_list.append(output)
        
        logger.info(f"Front back detection results {result_list}")
        
        if not result_list:
            return "unknown" 
        
        return max(result_list, key=result_list.count)
