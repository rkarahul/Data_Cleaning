import numpy as np
import cv2
from openvino.runtime import Core, get_version, PartialShape
import os
import traceback

class NormTvsClassifier:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = Core()
        available_devices = self.core.available_devices
        print(f"OpenVINO version: {get_version()}")
        print(f"Available devices: {available_devices}")
        
        if device.startswith("GPU") and not any(d.startswith("GPU") for d in available_devices):
            print("WARNING: GPU device requested but not available. Falling back to CPU.")
            device = "CPU"
        
        self.model = self.core.read_model(model_path)
        
        self.target_height = 224
        self.target_width = 224
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

    def classify_norm_tvs(self, image: np.ndarray) -> str:
        """Classifies the image as 'NORM' or 'TVS'."""
        try:
            preprocessed = self.preprocess_image(image)
            result = self.compiled_model({self.input_layer.any_name: preprocessed})
            predictions = result[self.output_layer][0]
            
            class_index = np.argmax(predictions)  # Get the class with the highest probability
            
            class_labels = ["NORM", "TVS"]
            return class_labels[class_index]
        
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            traceback.print_exc()
            return "unknown"

def process_images(folder_path, model_path, device="GPU"):
    detector = NormTvsClassifier(model_path, device)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.isfile(file_path):
            continue  # Skip directories
        
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read image: {file_name}")
            continue
        
        result = detector.classify_norm_tvs(image)
        print(f"Image: {file_name}, Prediction: {result}")

if __name__ == "__main__":
    folder_path = r"dataset\train\TVS"  # Change to your folder
    model_path = r"runs\classify\train\weights\best_openvino_model\best.xml"  # Change to your model path
    process_images(folder_path, model_path)
