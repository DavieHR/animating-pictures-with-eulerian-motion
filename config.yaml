import yaml
from easydict import EasyDict as edict
try:
    with open("config.yaml") as f:
        handle = edict(yaml.load(f))
        MODE = handle.mode
except Exception as e:
    print("no config.yaml")
    MODE = "fast"

def update(content):
    MODE = content


