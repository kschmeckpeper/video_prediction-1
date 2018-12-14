import itertools
import tensorflow as tf
from .base_dataset import VideoDataset
from .softmotion_dataset import SoftmotionVideoDataset


class ActionFreeVideoDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)
        self.state_like_names_and_shapes['images'] = '%d/env/image_view{}/encoded'.format(self.hparams.image_view), (48, 64, 3)

        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(ActionFreeVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,  #####
            time_shift=3,
            use_state=False,
            sdim=5,
            adim=4,
            image_view=0,
            compressed = True,
            append_touch=False
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(ActionFreeVideoDataset, self).parser(serialized_example)
        return state_like_seqs, action_like_seqs


    def num_examples_per_epoch(self):
        return 10000  ############
