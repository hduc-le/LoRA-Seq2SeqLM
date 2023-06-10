import os
import json
import pickle
import yaml

def save_pkl(save_object, save_file):
    with open(save_file, "wb") as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(load_file):
    with open(load_file, "rb") as f:
        output = pickle.load(f)
    return output

def load_json(path):
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    json_file.close()
    return data

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def write_jsonl(path, obj):
    with open(path, "w", encoding="utf-8") as writer:
        for item in obj:
            writer.write(json.dumps(item) + "\n")

def write_to_json(output_path, docs):
    with open(output_path, "w", encoding="utf-8") as fw:
        json.dump(docs, fw, ensure_ascii=False, indent=4)
    fw.close()

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def batch_to_device(batch, device, except_keys=[]):
    for key in batch:
        if key not in except_keys:
            batch[key] = batch[key].to(device)
    return batch

def read_config(path):
    # read yaml and return contents 
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
def print_config(d, indent=0):     
    for key, value in d.items():         
        if isinstance(value, dict):             
            print(' ' * indent + str(key))             
            print_config(value, indent+1)         
        else:             
            print(' ' * (indent+1) + f"{key}: {value}")

import logging
from colorlog import ColoredFormatter
def get_logger():
    global logger

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(message)s",
        #    datefmt='%H:%M:%S.%f',
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green,bold",
            "INFOV": "cyan,bold",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    ch.setFormatter(formatter)

    logger = logging.getLogger("rn")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # No duplicated handlers
    logger.propagate = False  # workaround for duplicated logs in ipython
    logger.addHandler(ch)
    return logger
