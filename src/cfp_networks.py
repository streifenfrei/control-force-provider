import sys
import os.path
import torch

OUTPUT_DIR = ""
LOG_FILE = None


def initialize(output_dir):
    global OUTPUT_DIR, LOG_FILE
    OUTPUT_DIR = output_dir
    LOG_FILE = os.path.join(OUTPUT_DIR, "cfp_rl.log")
