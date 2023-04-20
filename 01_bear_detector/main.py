from pathlib import Path

from fastai.torch_basics import GetAttr, add_props
from fastai.data.block import DataBlock
from fastai.basics import get_image_files, parent_label, RandomSplitter
from fastai.vision.data import ImageBlock, CategoryBlock
from fastai.vision.augment import Resize, RandomResizedCrop, aug_transforms
from fastai.vision.learner import vision_learner
from fastai.vision.all import resnet18, ResNet18_Weights
from fastai.metrics import error_rate
from fastai.interpret import ClassificationInterpretation
from fastai.vision.widgets import ImageClassifierCleaner
from fastai.learner import load_learner


PATH = Path(r"./01_bear_detector/data")


class DataLoaders(GetAttr):
    def __init__(self, *loaders):
        self.loaders = loaders

    def __getitem__(self, i):
        return self.loaders[i]

    train, valid = add_props(lambda i, self: self[i])


bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128),
)

dls = bears.dataloaders(PATH)

# Show a batch of images
# dls.valid.show_batch(max_n=4, nrows=1)

# Apply a random zoom to each image during each epoch
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(PATH)
# dls.train.show_batch(max_n=4, nrows=1, unique=True)

# Data augmentation
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(PATH)
# dls.train.show_batch(max_n=8, nrows=2, unique=True)

# Training
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms()
)
dls = bears.dataloaders(PATH)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=5)

learn.export(r"./01_bear_detector/bear_weights.pkl")

learn_inf = load_learner(r"./01_bear_detector/bear_weights.pkl")

learn_inf.predict(r"./01_bear_detector/data/black/00000000.jpg")
learn_inf.predict(r"./01_bear_detector/data/grizzly/00000000.jpg")
learn_inf.predict(r"./01_bear_detector/data/teddy/00000000.jpg")
