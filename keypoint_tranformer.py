# -*- coding: UTF-8 -*-
import pycolmap # pycolmap 占用的TLS最大,放在最前面 
# 原理参考: https://github.com/pytorch/pytorch/issues/2575#issue-254038499
from extract_features import normalize_keypoints, extractor_build, extract_img_feature
from torch.utils.data import DataLoader, Dataset
from os import path as osp
from pathlib import Path
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置Agg为后端
import matplotlib.pyplot as plt
import importlib
import warnings
import logging
import random
import torch
import copy
import time
import glob
import yaml
import cv2
import os
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = importlib.import_module('argparse').ArgumentParser(description="Extract feature and refine descriptor using neural network to find ship keypoint.")

    parser.add_argument(
        '--descriptor', type=str, default='SuperPoint+Boost-B-attlay3',
        help='descriptor to extract' )

    parser.add_argument(
        '--num_epochs', type=int, default=40,)

    parser.add_argument(
        '--train_ratio', type=float, default=1.0,
        help='The ratio of data used for training out of the training set' )

    parser.add_argument(
        '--batch_size', type=int, default=64,)

    parser.add_argument(
        '--num_workers', type=int, default=4,)

    parser.add_argument(
        '--print_interval', type=int, default=20,)

    parser.add_argument(
        '--eval_interval', type=int, default=1,)

    parser.add_argument(
        '--save_interval', type=int, default=10,)

    parser.add_argument(
        '--lr', type=float, default=1e-3,)

    parser.add_argument(
        '--warmup_step', type=int, default=20,)

    parser.add_argument(
        '--random_seed', type=int, default=0,)

    parser.add_argument(
        '--expand_piexl', type=int, default=5,)

    parser.add_argument(
        '--test_threshold', type=float, default=0.5,)

    parser.add_argument(
        '--test_image', type=str, default='',)
    
    parser.add_argument(
        '--test_images', nargs='*', default=[],)
    
    parser.add_argument(
        '--eval', action='store_true',)

    parser.add_argument(
        '--save_path', type=str, default='',)

    parser.add_argument(
        '--log_file', type=str, default='',)

    parser.add_argument(
        '--checkpoint', type=str, default='',)

    parser.add_argument(
        '--multiprocessing_context', type=str, default=None,)

    parser.add_argument(
        '--data_root', type=str, default='data/hrsid',)

    parser.add_argument(
        '--img_suffix', type=str, default='.png',)

    parser.add_argument(
        '--ann_suffix', type=str, default='.txt',)
    
    parser.add_argument(
        '--train_ann_file', nargs='*', default=['train/','val/'],)

    parser.add_argument(
        '--test_ann_file', nargs='*', default=['test/all','test/offshore','test/inshore'],)

    parser.add_argument(
        '--dataset_class',
        nargs='*',
        # default=['ore-oil','Cell-Container','Fishing','LawEnforce','Dredger','Container'], # [166, 89 , 288 , 25 , 263 , 2053]
        default=["ship"],
        help='Dataset classes list. Default is ["ship"].'
    )
    
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,)
    
    parser.add_argument(
        '--dataset_repeat', type=int, default=1,)
    
    parser.add_argument(
        '--positive_keypoint_repeat', type=int, default=1,)

    parser.add_argument(
        '--image_aug', action='store_true',)
    
    parser.add_argument(
        '--print', action='store_true')

    args = parser.parse_args()
    args.dataset_class = {item: index + 1 for index, item in enumerate(args.dataset_class)}    
    return args

def calculate_md5(file_path):

    hash_md5 = importlib.import_module('hashlib').md5()
    with open(file_path, "rb") as f:
        # 以块的方式读取文件，以防文件太大
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 定义舰船目标关键点检测模型（示例）
class ShipKeyPointsModel(nn.Module):
    def __init__(self, args,):
        super().__init__()
        FeatureBooster = importlib.import_module('FeatureBooster.featurebooster').FeatureBooster
        MLP = importlib.import_module('FeatureBooster.featurebooster').MLP

        self.device = args.device

        #读取Featurebooster的配置文件
        with open(str(Path(__file__).parent / "config.yaml"), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config[args.descriptor]
        # Model
        self.feature_booster = FeatureBooster(self.config)
        # load the model
        if os.path.isfile(args.feature_booster_pretrained):
            self.feature_booster.load_state_dict(torch.load(args.feature_booster_pretrained), strict =False)
            args.logger.info(f"feature_booster weights loaded from {args.feature_booster_pretrained}!")
        self.fc_out = MLP([self.config['output_dim'], self.config['output_dim']//2, len(args.dataset_class) + 1]) 
        self.to(args.device)
        self.args = args

    def forward(self, x):
        feat = self.feature_booster(x[...,self.config['keypoint_dim']:], x[...,:self.config['keypoint_dim']])
        x = self.fc_out(feat)
        return feat, nn.functional.softmax(x, dim=-1)

#旋转矩形框
def rotate_box_90_degrees(coords):
    x1, y1, x2, y2, x3, y3, x4, y4 = coords

    # 计算中心点
    cx = (x1 + x3) / 2
    cy = (y1 + y3) / 2

    # 将矩形平移到原点，旋转90度，然后平移回去
    rotated_coords = []
    for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
        x_prime = -y + cy + cx
        y_prime = x - cx + cy
        rotated_coords.extend([x_prime, y_prime])

    return rotated_coords

def load_txt_info_srsdd(txt_file, args):
    data_info = {}
    img_id = osp.split(txt_file)[1][:-4]
    data_info['img_id'] = img_id
    img_name = img_id + f'.{args.img_suffix.lstrip(".")}'
    data_info['file_name'] = img_name

    img_path = txt_file.replace(f'.{args.ann_suffix.lstrip(".")}',f'.{args.img_suffix.lstrip(".")}').replace('labels','images')
    data_info['img_path'] = img_path

    instances = []
    with open(txt_file) as f:
        s = f.readlines()
        for si in s[2:]:
            instance = {}
            bbox_info = si.split()
            instance['bbox_label'] = args.dataset_class[bbox_info[8]] if len(args.dataset_class) > 1 else 1
            instance['ignore_flag'] = 0
            instance['bbox'] = [float(i) for i in bbox_info[:8]]
            instances.append(instance)
    data_info['instances'] = instances
    return data_info

def load_txt_info_rsdd(txt_file, args):
    data_info = {}
    img_id = osp.split(txt_file)[1][:-4]
    data_info['img_id'] = img_id
    img_name = img_id + f'.{args.img_suffix.lstrip(".")}'
    data_info['file_name'] = img_name

    img_path = txt_file.replace(f'.{args.ann_suffix.lstrip(".")}',f'.{args.img_suffix.lstrip(".")}').replace('labels','images')
    data_info['img_path'] = img_path

    instances = []
    with open(txt_file) as f:
        s = f.readlines()
        for si in s:
            instance = {}
            bbox_info = si.split()
            instance['bbox_label'] = args.dataset_class[bbox_info[8]] if len(args.dataset_class) > 1 else 1
            instance['ignore_flag'] = 0
            instance['bbox'] = rotate_box_90_degrees([float(i) for i in bbox_info[2:]]) # 这里标注出的矩形框和真实中间差90度。可能是标注换换的问题。
            #人为旋转90度
            instances.append(instance)
    data_info['instances'] = instances
    return data_info

def load_txt_info_hrsid(txt_file, args):
    data_info = {}
    img_id = osp.split(txt_file)[1][:-4]
    data_info['img_id'] = img_id
    img_name = img_id + f'.{args.img_suffix.lstrip(".")}'
    data_info['file_name'] = img_name

    img_path = txt_file.replace(f'.{args.ann_suffix.lstrip(".")}',f'.{args.img_suffix.lstrip(".")}').replace('labelTxt','images')
    data_info['img_path'] = img_path

    instances = []
    with open(txt_file) as f:
        s = f.readlines()
        for si in s:
            instance = {}
            bbox_info = si.split()
            instance['bbox_label'] = args.dataset_class[bbox_info[8]]
            instance['ignore_flag'] = 0
            instance['bbox'] = [float(i) for i in bbox_info[:8]]
            instances.append(instance)
    data_info['instances'] = instances
    return data_info

def load_txt_info_sodaa(txt_file, args):
    dataset_class = ["airplane", "helicopter", "small-vehicle", "large-vehicle", "ship" ,"container","storage-tank","swimming-pool","windmill","ignore" ]
    json = importlib.import_module("json")
    data_info = {}
    with open(txt_file) as f:
        s = json.load(f)
        data_info['img_id'] = s['images']['file_name'][:-4]
        data_info['file_name'] = s['images']['file_name']
        data_info['img_path'] = osp.join(args.data_root,"Images", data_info['file_name'])
        data_info['image_shape'] = [s['images']['width'], s['images']['height']]
        instances = []
        for si in s['annotations']:
            instance = {}
            instance['bbox_label'] = args.dataset_class[dataset_class[int(si['category_id'])]]
            instance['bbox'] = [float(i) for i in si['poly']]
            instance['area'] = si['area']
            instance['ignore_flag'] = 0
            instances.append(instance)
        data_info['instances'] = instances
    return data_info

def get_keypoint_label(keypoints, data_info, args):
    bboxes = []
    bbox_label = []
    for instances in data_info['instances']:
        bboxes.append(np.array([(instances['bbox'][i], instances['bbox'][i + 1]) for i in range(0, len(instances['bbox']), 2)], dtype=np.int32))
        bbox_label.append(instances['bbox_label'])

    #创建和图像同样大小的空白区域，按照标注填充区域
    tmp = np.zeros(data_info['image_shape'], dtype=np.uint8)
    for box, label in zip(bboxes, bbox_label):
        cv2.fillPoly(tmp, np.array([box]), label)
    target = []
    mode = importlib.import_module('scipy.stats').mode
    for kp in keypoints:
        region = tmp[max(0, int(kp[1] - args.expand_piexl)):min(int(kp[1] + args.expand_piexl), data_info['image_shape'][0]),
                    max(0, int(kp[0] - args.expand_piexl)):min(int(kp[0] +args.expand_piexl), data_info['image_shape'][1])].reshape(-1)
        region = region[region.astype(np.bool_)]
        if region.size > 0:
            # 计算区域的众数
            mode_val = mode(region, axis=None)[0][0]
            target.append(mode_val)
        else:
            target.append(0)
    return np.array(target), bboxes

# 定义数据集（示例）
class ShipKeyPointsDataset(Dataset):
    def __init__(self, args, dataset_repeat = 1, pipeline = [], debug = False, **kwargs):
        super(ShipKeyPointsDataset, self).__init__()
        if len(pipeline):
            importlib.import_module('mmdet.utils').register_all_modules(init_default_scope=False)
            importlib.import_module('mmdet.utils').register_all_modules(init_default_scope=False)
        self.transform = importlib.import_module('mmengine.dataset').Compose(pipeline)
        self.debug = debug
        self.args = args
        with open(str(Path(__file__).parent / "config.yaml"), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config[args.descriptor]
        self.extractor = extractor_build(self.args.descriptor, device = args.device)
        if 'ann_file' in kwargs and kwargs['ann_file'] !='':
            # train case
            ann_dir = kwargs['ann_file']
            if isinstance(ann_dir, str):
                ann_dir = [ann_dir]
        else:
            ann_dir = []
        self.txt_files = []
        for path in ann_dir:
            self.txt_files.extend(glob.glob(osp.join(args.data_root, path, f"**/*.{args.ann_suffix.lstrip('.')}"), recursive=True))
        if (not args.eval) and (args.train_ratio<1):
            self.txt_files = random.sample(self.txt_files, int(np.ceil(len(self.txt_files)*args.train_ratio)))
        if (not args.eval) and (dataset_repeat > 1):
            self.txt_files = self.txt_files*dataset_repeat
            args.logger.info(f"The dataset located in {ann_dir} was duplicated {dataset_repeat} times!")

    def __len__(self):
        return len(self.txt_files)

    def load_data_info(self, idx):
        txt_file = self.txt_files[idx]
        load_txt_info_func = globals()['load_txt_info_' + self.args.data_root.rstrip('/').split('/')[-1]]
        return load_txt_info_func(txt_file, self.args)

    def __getitem__(self, idx):
        data_info = self.load_data_info(idx)
        bboxes = []
        if (not self.args.eval) and (len(self.transform.transforms)):
            data = self.transform(data_info)
            data_instance = data['data_samples'].gt_instances
            if (len(data_info['instances'])>0) and (len(data_instance.labels)<1):
                image = cv2.cvtColor(cv2.imread(data_info['img_path']), cv2.COLOR_BGR2RGB)
                data_info['image_shape'] = image.shape[:2]
            else:            
                image = data['inputs'].cpu().numpy().transpose(1, 2, 0)
                data_info['image_shape'] = image.shape[:2]
                data_info['instances'] = [] 
                for box_id in range(len(data_instance.labels)):
                    instance = {}
                    instance['bbox_label'] = int(data_instance.labels[box_id])
                    instance['bbox'] = list(data_instance.bboxes.vertices[box_id].reshape(-1).cpu().numpy())
                    data_info['instances'].append(instance)
            del data
        else:
            image = cv2.cvtColor(cv2.imread(data_info['img_path']), cv2.COLOR_BGR2RGB)
            data_info['image_shape'] = image.shape[:2]

        # 提取关键点和描述子
        try:
            keypoints, descriptors, image = extract_img_feature(self.args.descriptor, image, self.extractor)
        except BaseException as e:
            message = f"{data_info['img_path']} failed to extract img_feature!!!\n {e}"
            print(message)
            with open(self.args.log_file, "a") as file:   
                file.write(message+'\n')
            keypoints = np.array([])
        if keypoints.size == 0:
            message = f"{data_info['img_path']} has no keypoint founded with {self.args.descriptor}"
            print(message)
            with open(self.args.log_file, "a") as file:   
                file.write(message+'\n')
            return torch.zeros([2, self.config['keypoint_dim'] + self.config['descriptor_dim'] + 2], dtype = torch.float32, requires_grad = False).float(), data_info['img_path']
        else:
            target, bboxes = get_keypoint_label(keypoints, data_info, self.args)
            
            if (not self.args.eval) and (self.args.positive_keypoint_repeat>1) and (np.random.rand() > 0.5): # 将正样本的关键点特征进行重复，增加正样本数目
                keypoints = np.concatenate([keypoints, np.tile(keypoints[target.astype(np.bool_)], (self.args.positive_keypoint_repeat,1))], axis=0)
                descriptors = np.concatenate([descriptors, np.tile(descriptors[target.astype(np.bool_)], (self.args.positive_keypoint_repeat,1))], axis=0)
                target = np.concatenate([target, np.tile(target[target.astype(np.bool_)], self.args.positive_keypoint_repeat)], axis=0)

            # visualization
            if self.debug:
                print(f"VISUALIZATION: {data_info['img_path']}")
                kps = np.array([cv2.KeyPoint(*kp) for kp in keypoints])
                image = cv2.drawKeypoints(image, kps[target.astype(np.bool_)], None, color=(255,0,0,))
                image = cv2.drawKeypoints(image, kps[~(target.astype(np.bool_))], None, color=(0,0,255))
                image = cv2.polylines(image, bboxes, isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(f"vis_dir/{data_info['img_id']}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
            # boosted the descriptor using trained model
            keypoints = normalize_keypoints(keypoints, image.shape).astype(np.float32)
            if 'orb' in self.args.descriptor.lower():
                descriptors = np.unpackbits(descriptors, axis=1, bitorder='little').astype(np.float32)
                descriptors = descriptors * 2.0 - 1.0
            # 最后的全一是为了区分对齐batch的padding数据
            result = torch.from_numpy(np.concatenate([keypoints, descriptors, target.reshape(-1, 1), np.ones([len(target),1])], axis=-1))
            if (not self.args.eval) and (np.random.rand() > 0.5):
                result = result[torch.randperm(len(result))]
            result.requires_grad = False
            result = result.float()
            return result, data_info['img_path']

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_metric(all_labels, all_output, args):
    accuracy_score = importlib.import_module('sklearn.metrics').accuracy_score
    recall_score = importlib.import_module('sklearn.metrics').recall_score
    precision_score = importlib.import_module('sklearn.metrics').precision_score
    precision_recall_curve = importlib.import_module('sklearn.metrics').precision_recall_curve
    average_precision_score = importlib.import_module('sklearn.metrics').average_precision_score
    f1_score = importlib.import_module('sklearn.metrics').f1_score
    if isinstance(all_output, torch.Tensor):
        if all_output.requires_grad:
            all_output = all_output.detach()
        all_output = all_output.cpu().numpy()
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.cpu().numpy()

    metrics_per_class = {}
    Average_Precision_Curve = {}
    avg_metrics = {}
  
    dataset_class = args.dataset_class.copy()
    if len(args.dataset_class) > 1:
        dataset_class['foreground'] = 0

    all_predict = np.zeros_like(all_output, dtype=bool)
    np.put_along_axis(all_predict, np.argmax(all_output, axis=-1).reshape(-1, 1), True, axis=-1)
     
    for class_name, idx in dataset_class.items():
        # 为当前类别准备标签和预测
        if class_name in 'foreground':
            class_labels = (all_labels != idx).astype(int)
            class_output = ~all_predict[:, 0]
            ap = average_precision_score(class_labels, 1-all_output[:, idx])
            precisions, recalls, _ = precision_recall_curve(class_labels, 1-all_output[:, idx])
        else:
            class_labels = (all_labels == idx).astype(int)
            class_output = all_predict[:, idx]
            ap = average_precision_score(class_labels, all_output[:, idx])
            precisions, recalls, _ = precision_recall_curve(class_labels, all_output[:, idx])
        # 计算指标
        accuracy = accuracy_score(class_labels, class_output)
        precision = precision_score(class_labels, class_output)
        recall = recall_score(class_labels, class_output)
        f1 = f1_score(class_labels, class_output)

        metrics_per_class[class_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1,
            'Average_Precision': ap,
        }
        Average_Precision_Curve[class_name] = {'Precision': precisions,'Recall': recalls}
        metrics_str = ', '.join(f"{metric}: {value:.4f}" for metric, value in metrics_per_class[class_name].items())
        args.logger.info(f"Metrics for {class_name}: {metrics_str}")

    if len(args.dataset_class)>1:    
        # 计算平均值
        for metric in metrics_per_class[list(args.dataset_class.keys())[0]].keys():
            avg_metrics[metric] = np.mean([class_metrics[metric] for class_metrics in metrics_per_class.values()])
        metrics_str = ', '.join(f"{metric}: {value:.4f}" for metric, value in avg_metrics.items())
        args.logger.info(f"Average metrics: {metrics_str}")
        
        if (not ( os.path.isfile(args.test_image) or (len(args.test_images)>0) )):
            for class_name, idx in dataset_class.items():
                plt.plot(Average_Precision_Curve[class_name]['Recall'], 
                        Average_Precision_Curve[class_name]['Precision'], label=class_name, color=args.color[idx % len(args.color)])
            # 设置图例和标签
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            if (args.eval): 
                PR_curve_path = 'work_dirs/' + f'{args.save_path.split("/")[-1][:-4]}_PR_curve_eval.png'
            else:
                PR_curve_path = 'work_dirs/' + f'{args.save_path.split("/")[-1][:-4]}_PR_curve_train.png'
            plt.savefig(PR_curve_path, bbox_inches='tight', dpi=300)
            plt.close('all') 
            args.logger.info(f"PR_curve has been saved to {PR_curve_path}:")        
        return avg_metrics, Average_Precision_Curve['foreground']
    return metrics_per_class[list(args.dataset_class.keys())[0]], Average_Precision_Curve[list(args.dataset_class.keys())[0]] 

def evaluate(model, eva_loader, args):
    model.eval()

    all_output = torch.tensor([], device=args.device)
    all_labels = torch.tensor([], device=args.device)

    with torch.no_grad():
        for i, (data, img_paths) in enumerate(eva_loader):
            data = data.to(args.device)
            _, outputs = model(data[:,:,:-2])
            valid = data[:,:,-1].reshape(-1).bool()
            all_output = torch.cat([all_output, outputs.view(-1,len(args.dataset_class)+1)[valid,:]], dim=0)
            all_labels = torch.cat([all_labels, data[:,:,-2].long().reshape(-1)[valid]], dim=0)

            if (i + 1) % args.print_interval == 0:
                args.logger.info(f"Epoch(test) : [{i + 1}/{len(eva_loader)}]")

    return get_metric(all_labels, all_output, args)

def test(model, test_image, args):
    model.eval()
    extractor = extractor_build(args.descriptor)
    keypoints, descriptors, image = extract_img_feature(args.descriptor, cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB), extractor)

    load_txt_info_func = globals()['load_txt_info_' + args.data_root.rstrip('/').split('/')[-1]]
    re = importlib.import_module("re")
    txt_file = re.compile(r'images?', re.IGNORECASE).sub('**', test_image.replace(f'.{args.img_suffix.lstrip(".")}', f".{args.ann_suffix.lstrip('.')}"))
    txt_file = glob.glob(txt_file, recursive=True)[0] 
    data_info = load_txt_info_func(txt_file, args)
    data_info['image_shape'] = image.shape[:2]
    labels, bboxes = get_keypoint_label(keypoints, data_info, args)
    
    kps = np.array([cv2.KeyPoint(*kp) for kp in keypoints])
    # boosted the descriptor using trained model
    keypoints = normalize_keypoints(keypoints, image.shape).astype(np.float32)
    if 'orb' in args.descriptor.lower():
        descriptors = np.unpackbits(descriptors, axis=1, bitorder='little').astype(np.float32)
        descriptors = descriptors * 2.0 - 1.0
    with torch.no_grad():
        output = model(torch.from_numpy(np.concatenate([keypoints, descriptors,], axis=-1)).to(args.device).float())[1].cpu().numpy()

    metric_dict, PR_dict = get_metric(labels, output, args)
    # predict = (1-output[...,0]) > args.test_threshold
    if len(args.dataset_class) < 2:
        predict = (output[:,-1]>args.test_threshold).astype(np.int32)
    else:
        predict = np.argmax(output, axis=-1)
    dataset_class = dict()
    for class_name, idx in args.dataset_class.items():
        dataset_class[str(idx)] = class_name
        index = (predict==idx)
        image = cv2.drawKeypoints(image, kps[index&(labels==idx)], None, color=hex_to_rgb(args.color[idx % len(args.color)]),)
        image = cv2.drawKeypoints(image, kps[index&(labels!=idx)], None, color=hex_to_rgb("#40E0D0"),)
        # image = cv2.drawKeypoints(image, kps[(~predict)&(labels)], None, color=(0,0,255)) # Aqua蓝色 漏检
        # image = cv2.drawKeypoints(image, kps[predict&labels], None, color=(0,0,255,),) # 黄色 正确预测(正样本)
        # image = cv2.drawKeypoints(image, kps[(~predict)&(~labels)], None, color= (0, 255, 0) ) # 绿色 正确预测(负样本)
        # image = cv2.drawKeypoints(image, kps[(label)], None, color=(255,0,0,))
        # image = cv2.drawKeypoints(image, kps[~predict], None, color=(0,255,0))
    
    # 判断每个标注框里面是否有正确预测的关键点    
    tmp = np.zeros(data_info['image_shape'], dtype=np.uint8)
    for i, kp in enumerate(kps):
        tmp[max(0, int(kp.pt[1] - args.expand_piexl)):min(int(kp.pt[1] + args.expand_piexl), data_info['image_shape'][0]),
            max(0, int(kp.pt[0] - args.expand_piexl)):min(int(kp.pt[0] +args.expand_piexl), data_info['image_shape'][1])]= predict[i]
    mode = importlib.import_module('scipy.stats').mode
    for instance in data_info['instances']:
        color=hex_to_rgb(args.color[instance['bbox_label'] % len(args.color)])
        box = np.array([(instance['bbox'][i], instance['bbox'][i + 1]) for i in range(0, len(instance['bbox']), 2)], dtype=np.int32)
        
        mask = np.zeros_like(tmp)
        cv2.fillPoly(mask, [box], 1)
        region = tmp[mask == 1].reshape(-1)
        region = region[region.astype(np.bool_)]
        box_predict = 0
        if region.size > 0:
            # 计算区域的众数
            box_predict = mode(region, axis=None)[0][0]
        if box_predict != instance['bbox_label']:
            color=hex_to_rgb("#40E0D0")
            
            # rect = cv2.minAreaRect(box)
            # angle = rect[2]
            # center = rect[0] 
            # text_scale = np.sqrt(rect[1][0] * rect[1][1]) / 50  # 根据需要调整这个缩放因子
            
            # 将文本放在旋转后的位置
            # rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, angle, 1), (image.shape[1], image.shape[0]))
            # cv2.putText(image, "Missed", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 1)
            # image = cv2.warpAffine(rotated_image, cv2.getRotationMatrix2D(center, -angle, 1), (image.shape[1], image.shape[0])) 
            
        image = cv2.polylines(image, [box], isClosed=True, color=color , thickness=1)
        # assert args.dataset_class[cls_name] == instance['bbox_label'] 
        # cls_name = dataset_class[str(instance['bbox_label'])]
        
    save_path = f"{args.data_root.rstrip('/').split('/')[-1]}_keypoint_results/{args.descriptor}/vis/" + test_image.split('/')[-1]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return metric_dict, PR_dict

def keypoint_match(model, args, test_image1, test_image2):
    model.eval()
    extractor = extractor_build(args.descriptor)
    load_txt_info_func = globals()['load_txt_info_' + args.data_root.rstrip('/').split('/')[-1]]

    imgs = []
    kps = []
    descriptors = []
    labels = []
    for img_path in [test_image1, test_image2]:
        kp, des, img = extract_img_feature(args.descriptor, cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), extractor)
        re = importlib.import_module("re")
        txt_file = re.compile(r'images?', re.IGNORECASE).sub('**', img_path.replace(f'.{args.img_suffix.lstrip(".")}', f".{args.ann_suffix.lstrip('.')}"))
        txt_file = glob.glob(txt_file, recursive=True)[0]
        data_info = load_txt_info_func(txt_file, args)
        data_info['image_shape'] = img.shape[:2]
        label, bboxes = get_keypoint_label(kp, data_info, args)

        if 'orb' in args.descriptor.lower():
            des = np.unpackbits(des, axis=1, bitorder='little').astype(np.float32)
            des = des * 2.0 - 1.0
        kp_norm = normalize_keypoints(kp, img.shape).astype(np.float32)
        des, _ = model(torch.from_numpy(np.concatenate([kp_norm, des], axis=-1)).to(args.device).float())
        des = des.detach().cpu().numpy()
        imgs.append(img)
        kps.append(kp)
        descriptors.append(des)
        labels.append(label)
        
    border = 10
    # 为了使两个图像的关键点对齐，将第二个图像的关键点坐标加上第一个图像的宽度
    kps[1] = np.array([cv2.KeyPoint(kp_i[0] + border + imgs[0].shape[1], kp_i[1], kp_i[2]) for kp_i in kps[1]])
    kps[0] = np.array([cv2.KeyPoint(kp_i[0] , kp_i[1], kp_i[2]) for kp_i in kps[0]])
    
    #计算两个图像的关键点之间的距离，找到匹配的关键点
    imgs[1] = cv2.copyMakeBorder(imgs[1], 0, 0, border, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    dis = torch.cdist(torch.from_numpy(descriptors[0]),torch.from_numpy(descriptors[1]))
    matches_AB = torch.argmin(dis,dim=-1).cpu().numpy()
    matches_BA = torch.argmin(dis,dim=0).cpu().numpy()
    
    #组合两个图像
    h1, w1 = imgs[0].shape[:2]
    h2, w2 = imgs[1].shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    img_matches = np.zeros((height, width, 3), dtype="uint8")
    img_matches[:h1, :w1, :] = imgs[0]
    img_matches[:h2, w1:(w1 + w2), :] = imgs[1]


    # 画出匹配关键点
    for kp_i, label_i in zip(kps, labels):
        for kp_j, label_j in zip(kp_i, label_i):
            pt2 = tuple(np.round(kp_j.pt).astype(int))
            cv2.circle(img_matches, pt2, 5, color = hex_to_rgb("#40E0D0") if label_j>0 else hex_to_rgb("#FF0000"), thickness = 1) # if label_j>0 else hex_to_rgb("#FF0000")
    def draw_matches(img, kps, matches, color = hex_to_rgb("#FFFF00"), inverse = False):
        if inverse:
            color = hex_to_rgb("#00FF00")
        matches = matches[labels[int(inverse)].astype(np.bool_)] 
        for src,des,idx in zip(kps[int(inverse)][labels[int(inverse)].astype(np.bool_)], kps[1-int(inverse)][matches], matches):
            pt1 = tuple(np.round(src.pt).astype(int))
            pt2 = tuple(np.round(des.pt).astype(int))
            if labels[1-int(inverse)][idx]:
                cv2.arrowedLine(img, pt1, pt2, color , thickness = 1, tipLength=0.01)
            else:
                cv2.arrowedLine(img, pt1, pt2, hex_to_rgb("#FF0000"), thickness = 1, tipLength=0.01)
    draw_matches(img_matches, kps, matches_AB)
    draw_matches(img_matches, kps, matches_BA, inverse = True)

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = bf.match(descriptors[0], descriptors[1])
    # matches = sorted(matches, key=lambda x: x.distance, reverse = True)
    # img_matches = cv2.drawMatches(imgs[0], kps[0], imgs[1], kps[1], matches, None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    save_path = f"{args.data_root.rstrip('/').split('/')[-1]}_keypoint_results/{args.descriptor}/kp_matched/{test_image1.split('/')[-1][:-4]}vs{test_image2.split('/')[-1]}"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))        
    cv2.imwrite(save_path, cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR))
    args.logger.info(f"result saved to {save_path}")
    return img_matches

def worker_init_fn(worker_id, group, args):
    # torch.cuda.set_device(worker_id) 指定数加载设备
    torch.cuda.manual_seed_all(worker_id)

def custom_collate_fn(batch):
    rnn_utils = importlib.import_module('torch.nn.utils.rnn')
    results = [item[0] for item in batch]  # 提取每个样本的result
    img_paths = [item[1] for item in batch]  # 提取每个样本的img_path
    padded_results = rnn_utils.pad_sequence(results, batch_first=True, padding_value=0)
    return padded_results, img_paths

def train(model, args):
    CosineAnnealingWarmRestarts = importlib.import_module('torch.optim.lr_scheduler').CosineAnnealingWarmRestarts
    LinearLR = importlib.import_module('torch.optim.lr_scheduler').LinearLR
    MultiStepLR = importlib.import_module('torch.optim.lr_scheduler').MultiStepLR
    ChainedScheduler = importlib.import_module('torch.optim.lr_scheduler').ChainedScheduler
    clip_grad_norm_ = importlib.import_module('torch.nn.utils').clip_grad_norm_
    optimizer_class = importlib.import_module('torch.optim').AdamW
    partial = importlib.import_module('functools').partial
    train_pipeline = []
    #数据增强
    if args.image_aug:
        train_pipeline = [
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='mmrotate.ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmrotate.RandomRotate',
                prob=0.5,
                angle_range=180,
                rotate_type='mmrotate.Rotate'),
            dict(
                type='mmdet.RandomFlip',
                prob=0.75,
                direction=['horizontal', 'vertical', 'diagonal']),
            dict(
                type='mmdet.RandomAffine',),
            dict(
                type='mmdet.PhotoMetricDistortion',),

            dict(
                type='mmrotate.ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='qbox')),
            dict(type='mmdet.PackDetInputs', meta_keys=())]
    train_dataset = ShipKeyPointsDataset(args, ann_file = args.train_ann_file, dataset_repeat = args.dataset_repeat, pipeline = train_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=custom_collate_fn, 
                              # worker_init_fn=partial(worker_init_fn, group='train', args = args),
                              # pin_memory=True, persistent_workers = True,
                              multiprocessing_context=args.multiprocessing_context)
    eva_loader = {}
    for ann_file in args.test_ann_file:
        eval_args = copy.deepcopy(args)
        eval_args.eval = True
        eva_dataset = ShipKeyPointsDataset(args = eval_args, ann_file = ann_file)
        eva_loader[ann_file] = DataLoader(eva_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=custom_collate_fn, 
                                # worker_init_fn=partial(worker_init_fn, group='eval', args = args), 
                                # pin_memory=True, persistent_workers = True,
                                multiprocessing_context=args.multiprocessing_context)
    if args.print:
        outputs = importlib.import_module('mmengine.analysis').get_model_complexity_info(
            model,
            input_shape=None,
            inputs=train_dataset.__getitem__(0)[0][:,:-2].float().to(args.device),  # the input tensor of the model
            show_table=True,  # show the complexity table
            show_arch=False)  # show the complexity arch
        for k, v in outputs.items():
            args.logger.info(f"{k}: {v}")
    # 定义损失函数和优化器
    weight = torch.softmax(1/torch.tensor([100000.0,  166, 89 , 288 , 25 , 263 , 2053], device = args.device),dim=-1) # [100000.0,  166, 89 , 288 , 25 , 263 , 2053]

    CenterLoss = getattr(importlib.import_module("CenterLoss"), "CenterLoss")
    loss_weight = 20
    centerloss = CenterLoss(len(args.dataset_class) + 1, model.config['output_dim']).to(args.device)
    nllloss = nn.NLLLoss().to(args.device)
    param_groups = [
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': centerloss.parameters(), 'lr': args.lr, 'weight_decay': 1e-4}
    ]
    optimizer = optimizer_class(param_groups) # 
    # warmup_scheduler = LinearLR(optimizer, start_factor=1.0 / 20, end_factor=1.0, total_iters=args.warmup_step) 
    if 'srsdd' in args.data_root: 
        scheduler = MultiStepLR(optimizer, milestones=[10, 60, 80, 90], gamma=0.5)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min = 1e-5)
    # scheduler = MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.5)
    # scheduler = ChainedScheduler([warmup_scheduler, step_scheduler])

    start_epoch = 0
    best_AP = 0.0
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        centerloss.load_state_dict(checkpoint['centerloss_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']),
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']+1
        best_AP = checkpoint['best_AP']
        args.logger.info(f'Continue training from epoch [{start_epoch}] !')

    # getGPUs = importlib.import_module('GPUtil').getGPUs
    # virtual_memory = importlib.import_module('psutil').virtual_memory
    # collect = importlib.import_module('gc').collect
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        model.train()
        for i, (data, img_paths)  in enumerate(train_loader):
            torch.cuda.empty_cache() # 在显存资源有限的情况下使用
            data = data.to(args.device)
            # collect() # 显式垃圾回收
            # memory = virtual_memory()
            # args.logger.info(f"总内存: {memory.total / (1024 ** 3):.2f} GB, 可用内存: {memory.available / (1024 ** 3):.2f} GB, 内存使用率: {memory.percent}%")
            # gpu = getGPUs()[0]
            # args.logger.info(f">>>before: GPU: {gpu.name}, 显存总量: {gpu.memoryTotal}MB, 显存使用: {gpu.memoryUsed}MB, 显存空闲: {gpu.memoryFree}MB")
            feat, outputs = model(data[:,:,:-2])
            # args.logger.info(f">>>after: GPU: {gpu.name}, 显存总量: {gpu.memoryTotal}MB, 显存使用: {gpu.memoryUsed}MB, 显存空闲: {gpu.memoryFree}MB")
            valid = data[:,:,-1].reshape(-1).bool()
            label_inc = data[:,:,-2].long().view(-1)[valid]
            feat = feat.view(-1,model.config['output_dim'])[valid,:]
            outputs = outputs.view(-1,len(args.dataset_class)+1)[valid,:]
            
            # label = nn.functional.one_hot(label_inc, num_classes= len(args.dataset_class) + 1).float()
            
            # if len(args.dataset_class) > 1: # soft_label
            #     softening_value = 0.1
            #     non_background_mask = (label_inc != 0).unsqueeze(dim=-1).repeat(1,len(args.dataset_class)+1)
            #     non_background_mask[...,0] = False
            #     label[non_background_mask] = label[non_background_mask]  * (1 - softening_value) + \
            #                                     (1 - label[non_background_mask]) * softening_value / (len(args.dataset_class)-1)  
                                                          
            # loss_cls = torch.mean(- 0.75*label * ((1-outputs)**2) *torch.log(outputs) - 0.25*((1-label) *(outputs**2)* torch.log(1-outputs)), dim = 0)
            # inter_cls_loss = torch.sum(loss_cls*weight)
            inter_cls_loss = nllloss(torch.log(outputs), label_inc)
            inner_cls_loss = centerloss(feat, label_inc)
            loss = loss_weight * inter_cls_loss + inner_cls_loss
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 35, 2)
            clip_grad_norm_(centerloss.parameters(), 35, 2)
            optimizer.step()
            del outputs, data, valid
            if (i + 1) % args.print_interval == 0:
                current_time = time.time()
                eta_seconds = (current_time - start_time) / (i+1) * ( (args.num_epochs - epoch ) * len(train_loader) - (i + 1))
                eta_str = str(int(eta_seconds // 3600)) + ':' + str(int((eta_seconds % 3600) // 60)) + ':' + str(int(eta_seconds % 60))
                # loss_str = ' '.join([f"{cls_name}:{loss_value:.4f}" for cls_name,loss_value in zip(['background'] + list(args.dataset_class.keys()), loss_cls)])
                args.logger.info(f"Epoch(train)"
                      f"[{epoch + 1}/{args.num_epochs}][{i + 1}/{len(train_loader)}]"
                      f"lr: {optimizer.param_groups[0]['lr']:.4e}  eta: {eta_str}  " 
                      f"time: {current_time - start_time:.4f}  inter_cls_loss: {inter_cls_loss:.4f} inner_cls_loss: {inner_cls_loss.item():.4f}")
        scheduler.step()
        if ((epoch+1) % args.eval_interval == 0) or (epoch == 0) or (epoch == args.num_epochs-1): #            
            for idx, ann_file in enumerate(args.test_ann_file): 
                args.logger.info(f"Epoch(test) {ann_file}:")
                metric_dict_all, PR_dict_all = evaluate(model, eva_loader[ann_file], args)
                if len(args.dataset_class) <= 1:
                    plt.plot(PR_dict_all['Recall'], PR_dict_all['Precision'], label=ann_file.strip("/").strip("\\").split('/')[-1], color=args.color[idx % len(args.color)])
            if len(args.dataset_class) <= 1:
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                PR_curve_path = 'work_dirs/' + f'{args.save_path.split("/")[-1][:-4]}_PR_curve_train.png'
                plt.savefig(PR_curve_path, bbox_inches='tight', dpi=300)
                plt.close('all') 
                args.logger.info(f"PR_curve has been saved to {PR_curve_path}:")      
                               
            # 检查是否有更好的模型，如果有，则保存权重
            if metric_dict_all['Average_Precision'] > best_AP:
                best_AP = metric_dict_all['Average_Precision']
                # 保存当前模型的权重
                torch.save(model.state_dict(), args.save_path)
                args.logger.info(f"Best model saved to  {args.save_path} with MD5 {calculate_md5(args.save_path)}, with {args.test_ann_file[-1]} image AP {best_AP:.4f}")
            if (epoch >= args.num_epochs-1):
                last_save_path = 'work_dirs/' + args.data_root.rstrip('/').split('/')[-1] + '_' + args.descriptor + f'_{args.train_ratio*100:.0f}' + '_last_model_weight.pth'
                torch.save(model.state_dict(), last_save_path)
                args.logger.info(f"Last model saved :{last_save_path}")

        if ((epoch+1) % args.save_interval == 0):
            for file_path in glob.glob(args.save_path[:-4] + '*_epoch.pth'):
                os.remove(file_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'centerloss_state_dict': centerloss.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_AP': best_AP,
                }, args.save_path[:-4] + f'_{epoch+1}_epoch.pth')

if __name__ == '__main__':
    args = parse_arguments()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    if ('alike' in args.descriptor.lower()) or ('superpoint' in args.descriptor.lower()) or ('hardnet' in args.descriptor.lower()) or ('sosnet' in args.descriptor.lower()):
        args.multiprocessing_context = 'spawn'

    args.feature_booster_pretrained = '' # Path(__file__).parent / str("FeatureBooster/models/" + args.descriptor + ".pth")
    pretrained_str = 'finetune' if os.path.isfile(args.feature_booster_pretrained) else 'scratch'
    args.save_path = args.save_path if len(args.save_path) else 'work_dirs/' + args.data_root.rstrip('/').split('/')[-1] + '_' + args.descriptor + f'_{args.train_ratio*100:.0f}' + f"_bs{args.batch_size}" + f'_best_model_weights_{pretrained_str}.pth'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.color = ['#00FF00', '#FF00FF', '#FF4500', '#0000FF', '#00FFFF','#FF1493', '#808000'] 
    model = ShipKeyPointsModel(args)
    
    # 创建日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))    
    logger.addHandler(console_handler)
    args.logger = logger
    if (not args.eval) and (not ( os.path.isfile(args.test_image) or (len(args.test_images)>0) )): # train
        args.log_file = args.log_file if len(args.log_file) else 'work_dirs/' + args.data_root.rstrip('/').split('/')[-1] + '_' + args.descriptor + f'_{args.train_ratio*100:.0f}' + f"_bs{args.batch_size}" + f'_dataset_repeat_{args.dataset_repeat}_positive_keypoint_repeat_{args.positive_keypoint_repeat}_{pretrained_str}.log'
        if not os.path.exists(os.path.dirname(args.log_file)):
            os.makedirs(os.path.dirname(args.log_file))
        # 创建文件处理程序
        file_handler = logging.FileHandler(args.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # 添加处理程序到日志器
        logger.addHandler(file_handler)
        logger.info(">>>>>=========================Start Train!===========================<<<<<")
        for k, v in vars(args).items():
            if k not in 'logger':
                logger.info(f"{k}: {v}")

        if args.print: 
            with open(__file__, 'r') as file:
                lines = file.readlines()
            with open(args.log_file, "a") as file:   
                for line in lines:
                    file.write(line[:-1]+'\n')
                file.write('\n')
        train(model, args)
        args.logger.info(f"model weights saved to {args.save_path} with MD5 {calculate_md5(args.save_path)}!")
        args.checkpoint = args.save_path
        args.eval = True

    model_weights_md5 = "init_md5"
    if os.path.isfile(args.checkpoint):
        if 'cpu' in args.device.type:
            model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc:storage), strict=False)
        else:
            model.load_state_dict(torch.load(args.checkpoint), strict=False)
        model_weights_md5 = calculate_md5(args.checkpoint)
        args.logger.info(f"model weights loaded from {args.checkpoint} with MD5 {model_weights_md5}!")
        model_weights_md5 = model_weights_md5[:5]
    
    for img in args.test_images:
        if os.path.isfile(img):
            args.logger.info(f"Epoch(test) - {img}:")
            metric_dict, PR_dict = test(model,img, args)
    for idx in range(len(args.test_images)):
        keypoint_match(model, args, args.test_images[idx%len(args.test_images)], args.test_images[(idx+1)%len(args.test_images)])
    if os.path.isfile(args.test_image):
        args.logger.info(f"Epoch(test) - {args.test_image}:")
        metric_dict, PR_dict = test(model, args.test_image, args)
    if args.eval:
        for idx, ann_file in enumerate(args.test_ann_file):
            args.logger.info(f"Epoch(test) {ann_file}:")
            eva_dataset = ShipKeyPointsDataset(args = args, ann_file = ann_file)
            eva_loader = DataLoader(eva_dataset, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, collate_fn=custom_collate_fn, 
                                    pin_memory=True, persistent_workers = True, multiprocessing_context=args.multiprocessing_context)
            metric_dict, PR_dict_all = evaluate(model, eva_loader, args)
            if len(args.dataset_class) <= 1:
                plt.plot(PR_dict_all['Recall'], PR_dict_all['Precision'], label=ann_file.strip("/").strip("\\").split('/')[-1], color=args.color[idx % len(args.color)])
        if len(args.dataset_class) <= 1:
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            PR_curve_path = 'work_dirs/' + f'{args.save_path.split("/")[-1][:-4]}_PR_curve_eval_{model_weights_md5}.png'
            plt.savefig(PR_curve_path, bbox_inches='tight', dpi=300)
            plt.close('all') 
            args.logger.info(f"PR_curve has been saved to {PR_curve_path}!")
