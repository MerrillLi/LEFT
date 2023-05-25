# Create a Timer
# Support for timing code snippets

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict


class PerfTimer:

    def __init__(self) -> None:
        
        self.elapsed = []
        self.start_times = []
        self.end_times = []

    def start(self):
        self.start_times.append(time.time())

    def end(self):
        self.end_times.append(time.time())
        self.elapsed.append(self.end_times[-1] - self.start_times[-1])
    
    def compute(self):
        return sum(self.elapsed) * 1000 / len(self.elapsed)