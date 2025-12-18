import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
def data_transforms(size):
    datatrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.CenterCrop(size),
    #transforms.CenterCrop(args.input_size),
    transforms.Normalize(mean=mean_train,
                         std=std_train)])
    return datatrans
def gt_transforms(size):
    gttrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(size),
    transforms.ToTensor()])
    return gttrans


class MVTec3dDataset(Dataset):
    def __init__(self, type, root):
        self.data = []
        self.root = root
        self.label_to_idx = {'bagel': '0', 'cable_gland': '1', 'carrot': '2', 'cookie': '3', 'dowel': '4', 'foam': '5',
                             'peach': '6', 'potato': '7', 'rope': '8', 'tire': '9'}
        self.image_size = (256, 256)
        self.load_data(type, root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['filename']
        target_filename = item['filename']
        label = item["label"]
        if item.get("maskname", None):
            mask = cv2.imread(self.root + item['maskname'], cv2.IMREAD_GRAYSCALE)
        else:
            if label == 0:  # good
                mask = np.zeros(self.image_size).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones(self.image_size)).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        prompt = ""
        source = cv2.imread(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)
        source = cv2.cvtColor(source, 4)
        target = cv2.cvtColor(target, 4)
        source = Image.fromarray(source, "RGB")
        target = Image.fromarray(target, "RGB")
        mask = Image.fromarray(mask, "L")
        # transform_fn = transforms.Resize(256, Image.BILINEAR)
        transform_fn = transforms.Resize(self.image_size)
        source = transform_fn(source)
        target = transform_fn(target)
        mask = transform_fn(mask)
        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        mask = transforms.ToTensor()(mask)
        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["clsname"]
        image_idx = self.label_to_idx[clsname]

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))

    def load_data(self, type, root):
        data = []
        for clsname in ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']:
            if type == 'train':
                json_filename = 'train.json'
                data.extend(self.get_json(os.path.join(root, clsname, 'train', 'good'), clsname, is_train = True))
            else:
                json_filename = 'test.json'
                test_dir_path = os.path.join(root, clsname, 'test')
                test_dirs = os.listdir(test_dir_path)
                for test_sub_dir in test_dirs:
                    data.extend(self.get_json(os.path.join(root, clsname, 'test', test_sub_dir, 'rgb'), clsname, is_train = False, test_sub_dir = test_sub_dir))
        self.data = data
        self.save_json(data, os.path.join(root, '..', '..', 'diad-main_2', 'training', 'MVTec_3d', json_filename)) # 更改

    def get_json(self, dir_path, clsname, is_train, test_sub_dir = None):
        data = []
        mask_arr = []
        for img_name in os.listdir(dir_path):
            # training dataset
            if is_train:
                img_path = os.path.join(clsname, 'train', 'good', img_name)
                label = 0
                label_name = 'good'
                info_dic = {"filename": img_path, "label": label, "label_name": label_name, "clsname": clsname}
            # testing dataset
            else:
                img_path = os.path.join(clsname, 'test', test_sub_dir, 'rgb', img_name)
                if test_sub_dir == 'good':
                    label = 0
                    label_name = 'good'
                    info_dic = {"filename": img_path, "label": label, "label_name": label_name, "clsname": clsname}
                else:
                    label = 1
                    label_name = 'defective'
                    mask_path = os.path.join(clsname, 'test', test_sub_dir, 'gt', img_name)
                    info_dic = {"filename": img_path, "label": label, "label_name": label_name, "maskname": mask_path, "clsname": clsname}
            data.append(info_dic)
        return data

    @staticmethod
    def save_json(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
