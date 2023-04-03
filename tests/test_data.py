import os
import sys

sys.path.insert(0, os.getcwd())
from option import TrainOption
from data import VideoFlow

if __name__ == '__main__':

    config = TrainOption().get_parser()
    video_data = VideoFlow("train", config)
    print(len(video_data))

    for data in video_data:
        for _i in data:
            print(_i.size())
            #pass

