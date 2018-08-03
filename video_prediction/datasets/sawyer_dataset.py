import itertools



from .base_dataset import VideoDataset

from .softmotion_dataset import SoftmotionVideoDataset





class SawyerVideoDataset(SoftmotionVideoDataset):

    def __init__(self, *args, **kwargs):

        VideoDataset.__init__(self, *args, **kwargs)

        self.state_like_names_and_shapes['images'] = '%d/env/image_view0/encoded', (48, 64, 3)

        if self.hparams.use_state:

            self.state_like_names_and_shapes['states'] = '%d/env/state', (5,)

            self.action_like_names_and_shapes['actions'] = '%d/policy/actions', (5,)

        self._check_or_infer_shapes()



    def get_default_hparams_dict(self):

        default_hparams = super(SawyerVideoDataset, self).get_default_hparams_dict()

        hparams = dict(

            context_frames=2,

            sequence_length=15,

            time_shift=3,

            use_state=True,
            adim = 5,
            sdim = 5,
            ignore_touch = False,
            image_view = 0

        )

        return dict(itertools.chain(default_hparams.items(), hparams.items()))