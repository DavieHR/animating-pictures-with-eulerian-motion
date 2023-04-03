import threading
import torch
import torch.nn


import util._parallel_pytorch as pp

class MultiTask(object):
    """MultiTask Class
       Attributes:
            threading_num: int, the number of your threadings 
            task_list: list, the list your task is included.
            task_batches: list, the batches of your splited task.
    """
    def __init__(self, 
                 threading_num,
                 task_list):
        self.threading_num = threading_num
        self.task_batches = self.split_task(task_list)

    def split_task(self, _task_list):
        """split your task into several batches
        """
        _task_batches = []
        _batch_num = len(_task_list) // self.threading_num
        for i in range(self.threading_num):
            _batch = _task_list[i * _batch_num : (i + 1) * _batch_num]
            _task_batches.append(_batch)
        if len(_task_list) % self.threading_num:
            _task_batches[-1].extend(_task_list[_batch_num * self.threading_num:])

        return _task_batches

    def __call__(self, 
                 kernel,
                 *kernel_args,**kernel_kwargs):
        threading_pool = [threading.Thread(target = kernel, args = (batch, *kernel_args)) for batch in self.task_batches]
 
        for _thread in threading_pool:
            _thread.start()

        for _thread in threading_pool:
            _thread.join()


class MultiGPU(object):
    """MultiGPU class
       Attributes:
            gpu_ids: list, gpu index list.
			type_pal: str, [pytorch, zh_pytorch_model, zh_pytorch_criterion]
    """
    def __init__(self, gpu_ids=[], type_par = "pytorch"):
        self.gpu_ids = gpu_ids
        from functools import partial
        if type_par == "pytorch":
            self.parallel_instance = partial(torch.nn.DataParallel, device_ids=self.gpu_ids)
        elif type_par == "zh_pytorch_model":
            self.parallel_instance = partial(pp.DataParallelModel, device_ids=self.gpus)
        elif type_par == "zh_pytorch_criterion":
            self.parallel_instance = partial(pp.DataParallelCriterion, device_ids=self.gpus)
        elif type_par == "DDP":
            self.parallel_instance = partial(torch.nn.parallel.DistributedDataParallel, device_ids=[0], output_device=0, broadcast_buffers=False, find_unused_parameters=True)

    def __call__(self, x):
        if len(self.gpu_ids) == 0:
            return x
        assert(torch.cuda.is_available())
        #x = convert_model(x)
        x.to(self.gpu_ids[0])
        x = self.parallel_instance(x)
        return x




