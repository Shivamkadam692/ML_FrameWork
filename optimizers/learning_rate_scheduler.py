class StepDecay:
    """Decrease learning rate by a factor every N epochs."""
    def __init__(self, drop_rate=0.5, epochs_drop=10):
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def __call__(self, epoch, initial_lr):
        return initial_lr * (self.drop_rate ** (epoch // self.epochs_drop))

class ExponentialDecay:
    """Decrease learning rate exponentially."""
    def __init__(self, decay_rate=0.01):
        self.decay_rate = decay_rate

    def __call__(self, epoch, initial_lr):
        return initial_lr * (1.0 / (1.0 + self.decay_rate * epoch))
