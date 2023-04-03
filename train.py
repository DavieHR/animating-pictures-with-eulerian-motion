from option import TrainOption
from model  import *
from util.helper import Glint_log as LOG

# Update:
# DDP Mode.
import torch.distributed as dist
def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

if __name__ == '__main__':
    config = TrainOption().get_parser()
    total_epoch = config.total_epoch
    start_epoch = config.start_epoch

    if config.distributed:
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    trainer = WarpNetTrainer(config)
    for i in range(start_epoch, total_epoch + 1):
        LOG("info")("Current Epoch is {}".format(i + 1))
        trainer.train(i)
    LOG("info")("Training Finish!")


    

