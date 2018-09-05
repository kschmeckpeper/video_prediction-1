import itertools
import tensorflow as tf
from .base_dataset import VideoDataset
from .softmotion_dataset import SoftmotionVideoDataset

class GelsightDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)
        self.state_like_names_and_shapes['images'] = '%d/img', (48, 64, 3)
        # if self.hparams.use_state:
        #     self.state_like_names_and_shapes['states'] = '%d/env/state', (self.hparams.sdim,)
        #     if self.hparams.append_touch:
        #         self.state_like_names_and_shapes['touch'] = '%d/env/finger_sensors', (1,)
        self.action_like_names_and_shapes['action'] = '%d/action', (3,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(GelsightDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=18,
            time_shift=3,
            use_state=False,
            sdim=5,
            adim=3,
            image_view=0,
            compressed = False,
            append_touch=False
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(GelsightDataset, self).parser(serialized_example)
        # if self.hparams.append_touch:
        #     touch_data = state_like_seqs.pop('touch')
        #     state_like_seqs['states'] = tf.concat((state_like_seqs['states'], tf.nn.sigmoid(touch_data)), axis = -1)
        return state_like_seqs, action_like_seqs
    
    def num_examples_per_epoch(self):
        return len(self.filenames)
