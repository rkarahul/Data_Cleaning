import onnxruntime as ort
import cv2
from torchvision import transforms
from PIL import Image

class FogDetector:
    def __init__(self, model_name='models/f og_classifier_new.onnx'):
        # ONNX Runtime session
        self.session = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])

        # transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),     
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.threshold = 0.5

    def process_image(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0) 
        return image.numpy()

    def detect(self, frame):
        """
        Detect fog in the given frame.
        """
        # Preprocessing
        input_frame = self.process_image(frame)

        outputs = self.session.run([self.output_name], {self.input_name: input_frame})
        prediction = outputs[0]

        # result
        if prediction[0] > self.threshold:
            return "Fog"
        return "Clear"
