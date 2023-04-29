from model_example import *
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda")
print(f'Using {device} for inference')

model = ClipBase32(num_classes=x)
path_to_model = 'xxx'

state_dict = torch.load(path_to_model, map_location=torch.device('cpu'))['state_dict']

# rename key
for key in list(state_dict.keys()):
    if key.startswith('backbone.'):
        new_key = key.replace('backbone.', '')
    else:
        new_key = key.replace('f.model.', 'classifier.')
    state_dict[new_key] = state_dict.pop(key)

msg = model.load_state_dict(state_dict,  strict=False)
model.eval().to(device)

items = []
predict_ = []
basewidth = 224

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
with torch.no_grad():
     img = Image.open(item_id).convert('RGB')
     img = preprocess(img).to(device)
     img = img.unsqueeze(0)

     out_model = model(img)
     output = torch.nn.functional.softmax(out_model, dim=1)
     index = output.data.cpu().numpy().argmax()
