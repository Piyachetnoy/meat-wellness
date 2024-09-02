import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
import numpy as np

class Net(nn.Module):
  def __init__(self, class_num):
    super(Net, self).__init__()
    self.feature = nn.Sequential(
          # ブロック1
          nn.Conv2d(1, 32, kernel_size=3, padding=(1,1), padding_mode="replicate"),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, padding=(1,1), padding_mode="replicate"),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          nn.Dropout(0.25),

          # ブロック2
          nn.Conv2d(64, 128, kernel_size=3, padding=(1,1), padding_mode="replicate"),
          nn.ReLU(),
          nn.Conv2d(128, 128, kernel_size=3, padding=(1,1), padding_mode="replicate"),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          nn.Dropout(0.25)
    )
    self.classifier = nn.Sequential(
        nn.Linear(32768, 512),
        nn.Dropout(0.6),
        nn.Linear(512, class_num)
    )
    self.flatten = nn.Flatten(start_dim=1)

  def forward(self, x):
    x = self.feature(x)
    x = self.flatten(x)
    x = self.classifier(x)
    return x

class Model():
  def __init__(self, path, num_class=2):
    # Netクラスのインスタンス化
    model = Net(num_class)
    model_path = 'modelnew.kpl'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # 前処理を行う
    input = self.pretreatment(path)
    #画像判別
    self.output = model(input)
    self.prob = nn.functional.softmax(self.output, dim=1) #確率に変換する
    self.result = torch.argmax(self.output).item()
    
  
  def pretreatment(self, img_path, new_size=(64, 64)):
    img = np.array(Image.open(img_path))

    weights = np.array([0.2989, 0.5870, 0.1140])
    gray_image = np.dot(img[...,:3], weights)

    pil_image = Image.fromarray((gray_image * 255).astype(np.uint8))
    pil_image_resized = pil_image.resize(new_size, Image.LANCZOS)
    resized_image = np.array(pil_image_resized).astype(np.float32) / 255.0

    transform = T.Compose([T.Normalize(mean=[0.5], std=[0.5])])
    image_tensor = torch.from_numpy(resized_image).unsqueeze(-3).unsqueeze(-4)
    image = transform(image_tensor)

    return image

if __name__ == "__main__":
    #model = Model(path="./page/steak.png")
    model = Model(path="./static/img/well.jpg")
    print(model.result)
    print(model.prob)