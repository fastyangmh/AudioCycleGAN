# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyCMUARCTICForVC, MyAudioFolder, AudioLightningDataModule
from typing import Callable, Any, Optional, Tuple
from glob import glob
from os.path import join
import random
import numpy as np


#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_class = MyAudioFolder
    return AudioLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        sample_rate=project_parameters.sample_rate,
        dataset_class=dataset_class)


#class
class MyAudioFolder(MyAudioFolder):
    def __init__(self,
                 root: str,
                 loader: Callable[[str], Any],
                 extensions=('wav', 'flac'),
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, loader, extensions, transform, target_transform)
        self.find_samples()

    def find_samples(self):
        samples = {}
        for c in self.classes:
            temp = []
            for ext in self.extensions:
                temp += glob(join(self.root, f'{c}/*.{ext}'))
            samples[c] = sorted(temp)
        assert len(samples[self.classes[0]]) == len(
            samples[self.classes[1]]
        ), f'the {self.classes[0]} and {self.classes[1]} dataset have difference lengths.\nthe length of {self.classes[0]} dataset: {len(samples[self.classes[0]])}\nthe length of {self.classes[1]} dataset: {len(samples[self.classes[1]])}'
        self.samples = samples

    def __len__(self) -> int:
        return sum([len(v) for v in self.samples.values()])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = []
        for c in self.classes:
            path = self.samples[c][index]
            s = self.loader(path=path)
            if self.transform:
                s = self.transform(s)
            sample.append(s)
        return sample[0], sample[1]

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(
                len(self.samples[self.classes[0]])),
                                  k=max_samples)
            self.samples = {
                k: np.array(v)[index]
                for k, v in self.samples.items()
            }


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample1 and sample2
    print('the dimension of sample1: {}'.format(x.shape))
    print('the dimension of sample2: {}'.format(y.shape))