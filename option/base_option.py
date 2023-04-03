import argparse
from util.helper import Glint_log as LOG

class BaseOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--snapshots", type=str, default=None, help="snapshots path.")        
        self.parser.add_argument("--gpu_nums", type=int, default=1, help="gpu num")        
        self.parser.add_argument("--tensorboard", type=str, default=None, help="tensorboard path.")
        self.parser.add_argument("--script_path", type=str, default=None, help="script path.")
        self.parser.add_argument("--resume_path", type=str, default=None, help="resume exp model path.")
        self.parser.add_argument("--flow_file_name", type = str, default = "average_flow.flo", help = "flow file name.")
        self.parser.add_argument("--norm_flow", action = 'store_true', help = "flow norm [-1,1] or not")
        self.parser.add_argument("--Gen_name", type=str, default="SymmetricWarpingNet", help = "Generator name.")
        self.parser.add_argument("--Dis_name", type=str, default="StyleGanDiscriminator", help = "Discriminator name.")
        self.parser.add_argument("--EMotion_name", type=str, default=None, help = "EMotion Net Name")
        self.parser.add_argument("--distributed", action = 'store_true', help = "DDP Training.")

    def get_parser(self):
        opt = self.parser.parse_args()
        self.print_options(opt)
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        LOG("info")(message)

