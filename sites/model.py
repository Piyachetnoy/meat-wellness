import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
import numpy as np
import os

class Net(nn.Module):
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),  # Adjust the input size to match the output from `feature`
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, class_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

class Model():
    def __init__(self, path, num_class=2, model_path = 'modelnew.kpl'):
        # Ensure the model path exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Instantiate and load the model
        self.model = Net(num_class)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set the model to evaluation mode

        # Preprocess the image
        input_image = self.pretreatment(path)
        input_tensor = input_image.unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            self.output = self.model(input_tensor)
            self.prob = nn.functional.softmax(self.output, dim=1)  # Convert to probabilities
            self.result = torch.argmax(self.output, dim=1).item()  # Get the predicted class index

    def pretreatment(self, img_path, new_size=(64, 64)):
        img = np.array(Image.open(img_path).convert('RGB'))

        weights = np.array([0.2989, 0.5870, 0.1140])
        gray_image = np.dot(img[...,:3], weights)

        pil_image = Image.fromarray((gray_image * 255).astype(np.uint8))
        pil_image_resized = pil_image.resize(new_size, Image.LANCZOS)
        resized_image = np.array(pil_image_resized).astype(np.float32) / 255.0

        transform = T.Compose([T.Normalize(mean=[0.5], std=[0.5])])
        image_tensor = torch.from_numpy(resized_image).unsqueeze(0)  # Add channel dimension
        image = transform(image_tensor)

        return image

if __name__ == "__main__":
    # Define path to the model and image
    image_path = mainpath + 'test/image2.jpeg'

    # Initialize and use the model
    model = Model(path=image_path, model_path=model_path)
    print(f"Predicted class: {model.result}")
    print(f"Class probabilities: {model.prob}")