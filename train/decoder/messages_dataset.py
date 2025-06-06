import torch, os, torchvision
from torchvision import transforms
from PIL import Image
import json
import random

def parse_jsonl_file(jsonl_file_path, read_limit=None):
    with open(jsonl_file_path, 'r') as file:
        all_infos = []
        keep_keys = ['images', 'messages']
        for line in file:
            try:
                sample = json.loads(line)
                sample = {key: sample[key] for key in keep_keys if key in sample}
                all_infos.append(sample)
            except Exception as e:
                print(f"Error: {e}")
                continue
            if read_limit and len(all_infos) >= read_limit:
                break
        return all_infos


class QwenVisual2Image(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = parse_jsonl_file(dataset_path)
        self.path = []
        self.ref_path = []
        self.user_messages = []
        for data in metadata:
            self.path.append(data['images'][1])
            self.ref_path.append(data['images'][0])
            self.user_messages.append(data['messages'])

        self.height = height
        self.width = width
        print('train dataset size:', len(self.path))
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def preprocess_image(self, image_path):
        rgb_image = Image.open(image_path).convert("RGB")
        
        target_height, target_width = self.height, self.width
        width, height = rgb_image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height*scale),round(width*scale)]
        rgb_image = torchvision.transforms.functional.resize(rgb_image,shape,interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(rgb_image)
        return image, rgb_image

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        image, rgb_image = self.preprocess_image(self.path[data_id])
        ref_image, ref_rgb_image = self.preprocess_image(self.ref_path[data_id])
        messages = self.user_messages[data_id]

        return {"messages": messages, "image": image, "target_rgb_image": rgb_image, "ref_rgb_image": ref_rgb_image}


    def __len__(self):
        return self.steps_per_epoch
