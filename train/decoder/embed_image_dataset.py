import torch, os, torchvision
from torchvision import transforms
from PIL import Image
import json


def parse_jsonl_file(jsonl_file_path, read_limit=None):
    with open(jsonl_file_path, 'r') as file:
        all_infos = []
        for line in file:
            try:
                sample = json.loads(line)
                sample['image'] = sample['image'][0]
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
        self.embed_paths = []
        for data in metadata:
            self.path.append(data['image'])
            self.embed_paths.append(data['embed_path'])
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


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.

        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height*scale),round(width*scale)]
        image = torchvision.transforms.functional.resize(image,shape,interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)

        embed = torch.load(self.embed_paths[data_id], weights_only=True)
        return {"embed": embed, "image": image}


    def __len__(self):
        return self.steps_per_epoch
