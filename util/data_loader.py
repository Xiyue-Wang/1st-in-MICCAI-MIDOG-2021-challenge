from pathlib import Path
import openslide
import numpy as np


class SlideContainer:

    def __init__(self, file: Path, annotations: dict, y, level: int = 0, width: int = 256, height: int = 256,
                 sample_func: callable = None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.annotations = annotations
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int = 0, y: int = 0):
        return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                               level=self.level, size=(self.width, self.height)))[:, :, :3]

    @property
    def shape(self):
        return self.width, self.height

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)
