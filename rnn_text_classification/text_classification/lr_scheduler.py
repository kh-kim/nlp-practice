import torch


class WarmupConstantLRSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
