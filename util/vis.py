import torchvision
from torch.utils.tensorboard import SummaryWriter
from util.helper import Glint_log as LOG


class Visor:
    def __init__(self, path):
        self.vis = SummaryWriter(path)
        self.visual = {}

    def __setitem__(self, name, value):
        self.visual[name] = value

    def gather(self, iteration, normalize):
        
        Text_dict = {"Format_Text": "Step [ {} ]:".format(iteration)}
        Image_dict = {}
        Video_dict= {}
        Hist_dict= {}

        for k, v in self.visual.items():
            if 'Image' in k:
                key = k.replace('Image', '')
                Image_dict[k] = v
            elif 'Text' in k:
                key = k.replace('Text', '')
                Text_dict[key] = v
                Text_dict["Format_Text"] += " {}: {}".format(key,v)
            elif 'Video' in k:
                key = k.replace('Video', '')
                Video_dict[k] = v
            elif 'Hist' in k:
                key = k.replace('Hist', '')
                Hist_dict[k] = v
        
        self._show_text(Text_dict, iteration)
        self._show_image(Image_dict, iteration, normalize)
        self._show_hist(Hist_dict, iteration)
        self._show_video(Video_dict, iteration)

    def _show_image(self, image_dict, iteration, normalize = True):
        for k, v in image_dict.items():
            grid = torchvision.utils.make_grid(v.detach(), normalize=normalize, scale_each=True)
            self.vis.add_image(k, grid, iteration)

    def _show_text(self, text_dict, iteration):
        if len(text_dict.keys()) <=1 :
            return 
        LOG("info")(text_dict["Format_Text"])
        for k, v in text_dict.items():
            if k != "Format_Text":
                self.vis.add_scalar(k, v, iteration)

    def _show_video(self, video_dict, iteration):
        for k, v in video_dict.items():
            self.vis.add_video(k, v, iteration)

    def _show_hist(self, activate_dict, iteration):
        for k, v in activate_dict.items():
            self.vis.add_histogram(k, v, iteration)

    def __call__(self, iteration, normalize = True):
        self.gather(iteration, normalize)
        self.visual.clear()

