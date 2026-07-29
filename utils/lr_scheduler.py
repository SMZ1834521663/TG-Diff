import math


class CosineWarmupLambda:
    def __init__(
        self,
        one_epoch_step: int,
        warm_up_epochs: int = 50,
        cosine_epochs: int = 50,
        decay_milestones=(1500, 3000),   #epoch
        whole_ratios=(0.8, 0.6),
        lr_min_ratio: float = 0.1,
    ):
        self.one_epoch_step = one_epoch_step
        self.warm_up_steps = warm_up_epochs * one_epoch_step
        self.steps_per_cycle = cosine_epochs * one_epoch_step

        self.decay_steps = [e * one_epoch_step for e in decay_milestones]
        self.whole_ratios = whole_ratios
        self.lr_min_ratio = lr_min_ratio

        assert len(self.decay_steps) == len(self.whole_ratios)

    def __call__(self, step: int) -> float:
        # -------- warmup --------
        if step < self.warm_up_steps:
            return 1.0

        # -------- cosine --------
        step -= self.warm_up_steps
        step_in_cycle = step % self.steps_per_cycle
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_cycle / self.steps_per_cycle))
        lr_ratio = self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine_decay

        # -------- whole decay --------
        whole_ratio = 1.0
        for decay_step, ratio in zip(self.decay_steps, self.whole_ratios):
            if step > decay_step:
                whole_ratio = ratio

        return lr_ratio * whole_ratio