
# ========================================================
# This module provide the class for process a single video
# ========================================================
# - The result would storage in the Memo
# - OutPut as logging
#

import logging
import gc




class VidAnalyzerPipe(object):
    def __init__(self, staticDetect, dynamicDetect, tracker):
        self.staticDetect = []
        self.dynamicDetect = []
        self.tracker = []
        self.pipeline=

    def add_StaticModel(self, staticModel):
        se
