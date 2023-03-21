import argparse

import cv2
import numpy as np
import torch
import json

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from gcs_upload import convert_gcs_address_to_csv, upload_blob
from util import extract_zip, make_zip


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path', defult='../config/infer/tinaface/tinaface_r50_fpn_bn.py')
    parser.add_argument('imgname', help='image file name', defult = '../data')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)


def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items


def infer_face():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    imgname = args.imgname
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)
    
    data_paths = make_dataset_folder(imgname)
    
    for img in data_paths:
        name = img
        data = dict(img_info=dict(name=img), img_prefix=None)

    data = data_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if device != 'cpu':
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data
    result = engine.infer(data['img'], data['img_metas'])[0]
    result_json = {}
    result_json[name] = result
    
    plot_result(result, imgfp, class_names, outfp='out.jpg')    
    
    #make sucess file
    out_files = make_dataset_folder(outfp)
    num_converted_files = len(out_files)
    make_zip(outfph, outfp)
    out_files = make_dataset_folder(out_path)
    threadworkers=None
   
    #upload sucess file & csv
    with ThreadPoolExecutor(threadworkers) as pool:
        gcs_addresses=list(pool.map(upload_blob,repeat(bucket_name), out_files))        
    df_path_su=convert_vaild_gcs_address_to_csv(gcs_addresses,out_path)
    csv_gcs_address=upload_blob(bucket_name, df_path_su)
    
    
    
    #plot_result(result, imgname, class_names)
    
    return result_json


if __name__ == '__main__':
    main()
