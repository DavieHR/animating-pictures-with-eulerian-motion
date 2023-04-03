from .base_option import BaseOption

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        self.parser.add_argument("--data_type", type=str, default="VideoFlow", help = "Dataset Type")
        self.parser.add_argument("--data_root", type=str, default=None, nargs = "+", help = "Dataset path")
        self.parser.add_argument("--batchSize", type=int, default=4, help = "per-gpu batch size.")
        self.parser.add_argument("--d_num", type=int, default=1, help = "Discriminator tune times.")
        self.parser.add_argument("--vis_interval", type=int, default=1, help = "visualization time interval.")
        self.parser.add_argument("--save_interval", type=int, default=1, help = "snapshot and eval time interval.")
        self.parser.add_argument("--start_epoch", type=int, default=0, help = "start epoch.")
        self.parser.add_argument("--total_epoch", type=int, default=100, help = "total epoch.")
        self.parser.add_argument("--learning_rate", type=float, default=1e-4, help = "training learning rate.")
        self.parser.add_argument("--learning_rate_coeffs", type = str, default = None, nargs = "+", help = "learning rate coeffs for every net")
        self.parser.add_argument("--lr_policy", type = str, default = 'step', help = "learning rate decent or uprise policy. (default is step.)")
        self.parser.add_argument("--lr_decay_iters", type=int, default = 20, help = "learning rate decay steps under step policy.")
        self.parser.add_argument("--crop_size", type=int, default = 256, help = "network actual feed-in size.")
        self.parser.add_argument("--data_worker_num", type=int, default = 8, help = "data worker num size.")
        self.parser.add_argument("--re_init_dataset", action = 'store_true', help = "re-initialize dataset when every epoch training ends.")
        self.parser.add_argument("--joint_training", action = 'store_true', help = "joint training.")
        self.parser.add_argument("--neg_data_path", type=str, default=None, help = "Dataset Type")
        self.parser.add_argument("--neg_prob", type=float, default=0.5, help = "negative sample probability.")
        self.parser.add_argument("--no_target_norm", action = 'store_true', help = "use target norm.")
        self.parser.add_argument("--mask_dir", type=str, default=None, nargs="+",  help = "mask dir")
        self.parser.add_argument("--sep_data", action = 'store_true', help = "using sepecific data.")

