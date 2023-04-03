from .base_option import BaseOption

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.parser.add_argument("--input", "-i", type=str, default=None, help = "folder or file.")
        self.parser.add_argument("--output", "-o", type=str, default=None, help = "folder or file.")
        self.parser.add_argument("--save_file_name", type=str, default = 'test.mp4', help = "file name if you do not specify a file name for output; Or the file name will be the postfix of your specified file.")
        self.parser.add_argument("--frames", type=int, default = 60, help = "generate frame count.")
        self.parser.add_argument("--verbose", action = 'store_true', help = "save all the intermediate results of your network")
        self.parser.add_argument("--use_cache", action = 'store_true', help = "use cache.")
        self.parser.add_argument("--use_fast_version", action = 'store_true', help = "fast version motion flow.")
        self.parser.add_argument("--separation_emotion_path", type=str, default = None, help = "(use other separated training model.)")
        self.parser.add_argument("--resolution", nargs='+', default = [512,512], help = "nnetwork input resolution.")
        self.parser.add_argument("--intensity", type=float, default = 1, help = "intensity of flow.")
