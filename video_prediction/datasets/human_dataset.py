import itertools
import tensorflow as tf
from .base_dataset import VideoDataset
from .softmotion_dataset import SoftmotionVideoDataset
import numpy as np


class HumanVideoDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)

        for i in self.hparams.image_view:
            img_name = 'images{}'.format(i)
            if i==0 or len(self.hparams.image_view) == 1:
                img_name = 'images'
            img_format = '%d/env/image_view{}/encoded'.format(i), (self.hparams.img_height, self.hparams.img_width, 3)
            self.state_like_names_and_shapes[img_name] = img_format
        self.action_like_names_and_shapes = {}
#        if self.hparams.use_state:
#            self.state_like_names_and_shapes['states'] = '%d/env/state', (self.hparams.sdim,)
#            if self.hparams.append_touch:
#                self.state_like_names_and_shapes['touch'] = '%d/env/finger_sensors', (1,)
#        self.action_like_names_and_shapes['actions'] = '%d/policy/actions', (self.hparams.adim,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(HumanVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            time_shift=3,
            img_width=64,
            img_height=48,
            use_state=True,
            sdim=5,
            adim=4,
            image_view=[0],
            compressed = True,
            append_touch=False,
            append_state=False,
            state_vec=[0.],
            use_action=False
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(HumanVideoDataset, self).parser(serialized_example)
 #       if self.hparams.append_touch:
 #           touch_data = state_like_seqs.pop('touch')
 #           state_like_seqs['states'] = tf.concat((state_like_seqs['states'], tf.nn.sigmoid(touch_data)), axis = -1)
 #       if self.hparams.use_state and self.hparams.append_state:
 #           extra_state = np.tile(np.array(self.hparams.state_vec).reshape((1, -1)), [self.hparams.sequence_length, 1])
 #           state_like_seqs['states'] = tf.concat([state_like_seqs['states'],
 #                                                  tf.convert_to_tensor(extra_state, dtype=tf.float32)], axis=1)
        state_shape = state_like_seqs['images'].shape
        # Subtract 1 because there is 1 more image than action
        action_like_seqs['actions'] = np.inf + tf.zeros([state_shape[0]-1, self.hparams.adim])
#        action_shape = action_like_seqs['actions'].shape
        print("state shape:", state_shape)
        print("Using action:", self.hparams.use_action)
        action_like_seqs['use_action'] = tf.constant(self.hparams.use_action, shape=[state_shape[0]-1, 1], dtype=tf.bool)
        return state_like_seqs, action_like_seqs
