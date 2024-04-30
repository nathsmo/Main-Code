from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from datetime import datetime
import time
import sys

print_grad = True

class PrintOut:
    def __init__(self, f=None, stdout_print=True):
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
            self.out_file.flush()

        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self, s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        duration = time.time() - start_time
        self.print_out(f"{s}, time {duration}s, {time.ctime()}.")
        return time.time()

    def print_grad(self, model, last=False):
        if print_grad:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.print_out(f'{name: <50} -- value: {param.data.norm():.12f} -- grad: {param.grad.data.norm():.12f}')
                else:
                    self.print_out(f'{name: <50} -- value: {param.data.norm():.12f}')
            self.print_out("-----------------------------------")
            if last:
                self.print_out("-----------------------------------")
                self.print_out("-----------------------------------")

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        for i, img in enumerate(images):
            self.writer.add_image(f'{tag}/{i}', img, step)

    def histo_summary(self, tag, values, step, bins=1000):
        self.writer.add_histogram(tag, values, step, bins=bins)

def gradient_clip(model, max_gradient_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
