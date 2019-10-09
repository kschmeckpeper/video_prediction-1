import itertools

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops, flow_ops
from video_prediction.models import VideoPredictionModel
from video_prediction.models import pix2pix_model, mocogan_model, spectral_norm_model
from video_prediction.ops import lrelu, dense, pad2d, conv2d, conv_pool2d, flatten, tile_concat, pool2d
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils
from video_prediction.losses import kl_loss, kl_loss_dist, js_loss, l2_loss

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


def create_legacy_encoder(inputs,
                          nz=8,
                          nef=64,
                          norm_layer='instance',
                          include_top=True):
    norm_layer = ops.get_norm_layer(norm_layer)

    with tf.variable_scope('h1'):
        h1 = conv_pool2d(inputs, nef, kernel_size=5, strides=2)
        h1 = norm_layer(h1)
        h1 = tf.nn.relu(h1)

    with tf.variable_scope('h2'):
        h2 = conv_pool2d(h1, nef * 2, kernel_size=5, strides=2)
        h2 = norm_layer(h2)
        h2 = tf.nn.relu(h2)

    with tf.variable_scope('h3'):
        h3 = conv_pool2d(h2, nef * 4, kernel_size=5, strides=2)
        h3 = norm_layer(h3)
        h3 = tf.nn.relu(h3)
        h3_flatten = flatten(h3)

    if include_top:
        with tf.variable_scope('z_mu'):
            z_mu = dense(h3_flatten, nz)
        with tf.variable_scope('z_log_sigma_sq'):
            z_log_sigma_sq = dense(h3_flatten, nz)
            z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
        outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    else:
        outputs = h3_flatten
    return outputs


def create_n_layer_encoder(inputs,
                           nz=8,
                           nef=64,
                           n_layers=3,
                           norm_layer='instance',
                           include_top=True):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(1, n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = nef * min(2**i, 4)
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=2, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    pooled = pool2d(rectified, rectified.shape[1:3].as_list(), padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])

    if include_top:
        with tf.variable_scope('z_mu'):
            z_mu = dense(squeezed, nz)
        with tf.variable_scope('z_log_sigma_sq'):
            z_log_sigma_sq = dense(squeezed, nz)
            z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
        outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    else:
        outputs = squeezed
    return outputs


def create_encoder(inputs, e_net='legacy', use_e_rnn=False, rnn='lstm', **kwargs):
    assert inputs.shape.ndims == 5
    batch_shape = inputs.shape[:-3].as_list()
    inputs = flatten(inputs, 0, len(batch_shape) - 1)
    unflatten = lambda x: tf.reshape(x, batch_shape + x.shape.as_list()[1:])

    if use_e_rnn:
        if e_net == 'legacy':
            kwargs.pop('n_layers', None)  # unused
            h = create_legacy_encoder(inputs, include_top=False, **kwargs)
            with tf.variable_scope('h4'):
                h = dense(h, kwargs['nef'] * 4)
        elif e_net == 'n_layer':
            h = create_n_layer_encoder(inputs, include_top=False, **kwargs)
            with tf.variable_scope('layer_%d' % (kwargs['n_layers'] + 1)):
                h = dense(h, kwargs['nef'] * 4)
        else:
            raise ValueError('Invalid encoder net %s' % e_net)

        if rnn == 'lstm':
            RNNCell = tf.contrib.rnn.BasicLSTMCell
        elif rnn == 'gru':
            RNNCell = tf.contrib.rnn.GRUCell
        else:
            raise NotImplementedError

        h = nest.map_structure(unflatten, h)
        for i in range(2):
            with tf.variable_scope('%s_h%d' % (rnn, i)):
                rnn_cell = RNNCell(kwargs['nef'] * 4)
                h, _ = tf.nn.dynamic_rnn(rnn_cell, h, dtype=tf.float32, time_major=True)
        h = flatten(h, 0, len(batch_shape) - 1)

        with tf.variable_scope('z_mu'):
            z_mu = dense(h, kwargs['nz'])
        with tf.variable_scope('z_log_sigma_sq'):
            z_log_sigma_sq = dense(h, kwargs['nz'])
            z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
        outputs = {'enc_zs_mu': z_mu, 'enc_zs_log_sigma_sq': z_log_sigma_sq}
    else:
        if e_net == 'legacy':
            kwargs.pop('n_layers', None)  # unused
            outputs = create_legacy_encoder(inputs, include_top=True, **kwargs)
        elif e_net == 'n_layer':
            outputs = create_n_layer_encoder(inputs, include_top=True, **kwargs)
        else:
            raise ValueError('Invalid encoder net %s' % e_net)

    outputs = nest.map_structure(unflatten, outputs)
    return outputs

def action_decoder_fn(encoded_actions, actions_shape, hparams=None, norm=None):
    with tf.variable_scope('action_decoder', reuse=tf.AUTO_REUSE):
        reshaped_encoded_actions = tf.reshape(encoded_actions, [-1, encoded_actions.shape[-1]])
        # print("reshaped encoded actions:", reshaped_encoded_actions.shape)
        if hparams.action_encoder_norm_layer is not None:
            norm_layer = ops.get_norm_layer(hparams.action_encoder_norm_layer)

        with tf.variable_scope('action_decoder_start'):
            out = dense(reshaped_encoded_actions, hparams.action_encoder_channels, kernel_init=None, bias_init=None)
            if norm is not None:
                out = norm_layer(out)
            out = tf.nn.relu(out)
        for i in range(hparams.action_encoder_layers):
            with tf.variable_scope('action_decoder_layer_{}'.format(i)):
                out = dense(out, hparams.action_encoder_channels, kernel_init=None, bias_init=None)
                if norm is not None:
                    out = norm_layer(out)
                out = tf.nn.relu(out)
        with tf.variable_scope('action_decoder_out'):
            action_reconstruction = dense(out, actions_shape[-1], kernel_init=None, bias_init=None)
            action_reconstruction = tf.reshape(action_reconstruction, [actions_shape[0], actions_shape[1], actions_shape[2]])

    return action_reconstruction

def action_encoder_fn(inputs, hparams=None, norm=None):
#    inputs['actions'] = tf.Print(inputs['actions'], [inputs['actions'][:, :9, :]], "First nine", summarize=-1)
    reshaped_actions = tf.reshape(inputs['actions'][:, :9, :], [-1, inputs['actions'].shape[-1]])
    if hparams.action_encoder_norm_layer is not None:
        norm_layer = ops.get_norm_layer(hparams.action_encoder_norm_layer)

    num_hidden_channels = hparams.action_encoder_channels

    with tf.variable_scope('action_encoder_start'):
        out = dense(reshaped_actions, num_hidden_channels, kernel_init=None, bias_init=None)
#        out = tf.Print(out, [out], "out after start", summarize=-1)
        if norm is not None:
            out = norm_layer(out)
        out = tf.nn.relu(out)
#        out = tf.Print(out, [out], "out after relu", summarize=-1)
    for i in range(hparams.action_encoder_layers):
        with tf.variable_scope('action_encoder_layer_{}'.format(i)):
            out = dense(out, num_hidden_channels, kernel_init=None, bias_init=None)
            if norm is not None:
                out = norm_layer(out)
            out = tf.nn.relu(out)
    with tf.variable_scope('action_encoder_out_mu'):
        action_mu = dense(out, hparams.encoded_action_size, kernel_init=None, bias_init=None)
        action_mu = tf.reshape(action_mu, [inputs['actions'].shape[0], -1, hparams.encoded_action_size])
    with tf.variable_scope('action_encoder_out_sigma_sq'):
        action_log_sigma_sq = dense(out, hparams.encoded_action_size)
#        action_log_sigma_sq = tf.clip_by_value(action_log_sigma_sq, -10, 10)
        action_log_sigma_sq = tf.reshape(action_log_sigma_sq, [inputs['actions'].shape[0], -1, hparams.encoded_action_size])


    outputs = {'action_log_sigma_sq': action_log_sigma_sq,
               'action_mu': action_mu}
    return outputs

def inverse_model_fn(inputs, hparams=None):
    images = inputs['images']
    image_pairs = tf.concat([images[:hparams.sequence_length - 1],
                             images[1:hparams.sequence_length]], axis=-1)
    if 'dx' in inputs and hparams.add_dx_to_inverse:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['dx'], axis=-2), axis=-2)], axis=-1)
    if 'da' in inputs and hparams.add_da_to_inverse:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['da'], axis=-2), axis=-2)], axis=-1)
    outputs = create_encoder(image_pairs,
                             e_net=hparams.inverse_model_net,
                             use_e_rnn=hparams.use_inverse_model_rnn,
                             rnn=hparams.rnn,
                             nz=hparams.encoded_action_size,
                             nef=hparams.nef,
                             n_layers=hparams.n_layers,
                             norm_layer=hparams.norm_layer)
    renamed_outputs = {'action_inverse_mu': outputs['enc_zs_mu'],
                       'action_inverse_log_sigma_sq': outputs['enc_zs_log_sigma_sq']}
    # print("Outputs for inverse model", renamed_outputs.keys())
    return renamed_outputs


def image_encoder_fn(inputs, hparams=None):
    images = inputs['images'][0]

    outputs = create_n_layer_encoder(images,
                                     nz=2*hparams.encoded_action_size,
                                     nef=hparams.nef,
                                     n_layers=hparams.n_layers,
                                     norm_layer=hparams.norm_layer)
    renamed_outputs = {'enc_image_mu': outputs['enc_zs_mu'],
                       'enc_image_log_sigma_sq': outputs['enc_zs_log_sigma_sq']}
    # print("Outputs for image encoder", renamed_outputs.keys())
    return renamed_outputs


def image_decoder_fn(inputs, hparams=None):
    raise NotImplementedError


def encoder_fn(inputs, hparams=None):
    images = inputs['images']


    image_pairs = tf.concat([images[:hparams.sequence_length - 1],
                             images[1:hparams.sequence_length]], axis=-1)


    if 'actions' in inputs:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['actions'], axis=-2), axis=-2)], axis=-1)
    outputs = create_encoder(image_pairs,
                             e_net=hparams.e_net,
                             use_e_rnn=hparams.use_e_rnn,
                             rnn=hparams.rnn,
                             nz=hparams.nz,
                             nef=hparams.nef,
                             n_layers=hparams.n_layers,
                             norm_layer=hparams.norm_layer)
    return outputs


def discriminator_fn(targets, inputs=None, hparams=None):
    outputs = {}
    if hparams.gan_weight or hparams.vae_gan_weight:
        _, pix2pix_outputs = pix2pix_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(pix2pix_outputs)
    if hparams.image_gan_weight or hparams.image_vae_gan_weight or \
            hparams.video_gan_weight or hparams.video_vae_gan_weight or \
            hparams.acvideo_gan_weight or hparams.acvideo_vae_gan_weight:
        _, mocogan_outputs = mocogan_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(mocogan_outputs)
    if hparams.image_sn_gan_weight or hparams.image_sn_vae_gan_weight or \
            hparams.video_sn_gan_weight or hparams.video_sn_vae_gan_weight:
        _, spectral_norm_outputs = spectral_norm_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(spectral_norm_outputs)
    return None, outputs


class DNACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(DNACell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams

        if self.hparams.where_add not in ('input', 'all', 'middle'):
            raise ValueError('Invalid where_add %s' % self.hparams.where_add)

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape
        scale_size = max(height, width)
        if scale_size == 256:
            self.encoder_layer_specs = [
                (self.hparams.ngf, False),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 8, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf, False),
                (self.hparams.ngf, False),
            ]
        elif scale_size == 128:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 8, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]

        elif scale_size == 64:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf * 4, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        elif scale_size == 32:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        else:
            raise NotImplementedError

        # output_size
        gen_input_shape = list(image_shape)
        if 'encoded_actions' in inputs:
            gen_input_shape[-1] += inputs['encoded_actions'].shape[-1].value
        elif 'actions' in inputs:
            # print("gen_input in cell:",gen_input_shape)
            # print("actions in cell", inputs['actions'].shape)
            gen_input_shape[-1] += inputs['actions'].shape[-1].value

        num_masks = self.hparams.last_frames * self.hparams.num_transformed_images + \
            int(bool(self.hparams.prev_image_background)) + \
            int(bool(self.hparams.first_image_background and not self.hparams.context_images_background)) + \
            (self.hparams.context_frames if self.hparams.context_images_background else 0) + \
            int(bool(self.hparams.generate_scratch_image))
        output_size = {
            'gen_images': tf.TensorShape(image_shape),
            'gen_inputs': tf.TensorShape(gen_input_shape),
            'transformed_images': tf.TensorShape(image_shape + [num_masks]),
            'masks': tf.TensorShape([height, width, 1, num_masks]),
        }
        if 'pix_distribs' in inputs:
            num_motions = inputs['pix_distribs'].shape[-1].value
            output_size['gen_pix_distribs'] = tf.TensorShape([height, width, num_motions])
            output_size['transformed_pix_distribs'] = tf.TensorShape([height, width, num_motions, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        if self.hparams.transformation == 'flow':
            output_size['gen_flows'] = tf.TensorShape([height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            output_size['gen_flows_rgb'] = tf.TensorShape([height, width, 3, self.hparams.last_frames * self.hparams.num_transformed_images])
        self._output_size = output_size

        # state_size
        conv_rnn_state_sizes = []
        conv_rnn_height, conv_rnn_width = height, width
        for out_channels, use_conv_rnn in self.encoder_layer_specs:
            conv_rnn_height //= 2
            conv_rnn_width //= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        for out_channels, use_conv_rnn in self.decoder_layer_specs:
            conv_rnn_height *= 2
            conv_rnn_width *= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        if self.hparams.conv_rnn == 'lstm':
            conv_rnn_state_sizes = [tf.nn.rnn_cell.LSTMStateTuple(conv_rnn_state_size, conv_rnn_state_size)
                                    for conv_rnn_state_size in conv_rnn_state_sizes]
        state_size = {'time': tf.TensorShape([]),
                      'gen_image': tf.TensorShape(image_shape),
                      'last_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'conv_rnn_states': conv_rnn_state_sizes}
        if 'zs' in inputs and self.hparams.use_rnn_z:
            rnn_z_state_size = tf.TensorShape([self.hparams.nz])
            if self.hparams.rnn == 'lstm':
                rnn_z_state_size = tf.nn.rnn_cell.LSTMStateTuple(rnn_z_state_size, rnn_z_state_size)
            state_size['rnn_z_state'] = rnn_z_state_size
        if 'pix_distribs' in inputs:
            state_size['gen_pix_distrib'] = tf.TensorShape([height, width, num_motions])
            state_size['last_pix_distribs'] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
        if 'states' in inputs:
            state_size['gen_state'] = inputs['states'].shape[2:]
        self._state_size = state_size

        ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        if self.hparams.schedule_sampling == 'none':
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif self.hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if self.hparams.schedule_sampling == 'inverse_sigmoid':
                k = self.hparams.schedule_sampling_k
                # print("Sampling k", k)
                # print("start step", self.hparams.schedule_sampling_steps)
                start_step = self.hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
                # prob = tf.Print(prob, [prob], "Sampling prob")
            elif self.hparams.schedule_sampling == 'linear':
                start_step, end_step = self.hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * batch_size, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[self.hparams.context_frames, batch_size])
        self.ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def zero_state(self, batch_size, dtype):
        init_state = super(DNACell, self).zero_state(batch_size, dtype)
        init_state['last_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        if 'pix_distribs' in self.inputs:
            init_state['last_pix_distribs'] = [self.inputs['pix_distribs'][0]] * self.hparams.last_frames
        return init_state

    def _rnn_func(self, inputs, state, num_units):
        if self.hparams.rnn == 'lstm':
            RNNCell = tf.contrib.rnn.BasicLSTMCell
        elif self.hparams.rnn == 'gru':
            RNNCell = tf.contrib.rnn.GRUCell
        else:
            raise NotImplementedError
        rnn_cell = RNNCell(num_units, reuse=tf.get_variable_scope().reuse)
        return rnn_cell(inputs, state)

    def _conv_rnn_func(self, inputs, state, filters):
        inputs_shape = inputs.get_shape().as_list()
        input_shape = inputs_shape[1:]
        if self.hparams.norm_layer == 'none':
            normalizer_fn = None
        else:
            normalizer_fn = ops.get_norm_layer(self.hparams.norm_layer)
        if self.hparams.conv_rnn == 'lstm':
            Conv2DRNNCell = BasicConv2DLSTMCell
        elif self.hparams.conv_rnn == 'gru':
            Conv2DRNNCell = Conv2DGRUCell
        else:
            raise NotImplementedError
        if self.hparams.ablation_conv_rnn_norm:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          reuse=tf.get_variable_scope().reuse)
            h, state = conv_rnn_cell(inputs, state)
            outputs = (normalizer_fn(h), state)
        else:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          normalizer_fn=normalizer_fn,
                                          separate_norms=self.hparams.norm_layer == 'layer',
                                          reuse=tf.get_variable_scope().reuse)
            outputs = conv_rnn_cell(inputs, state)
        return outputs

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        downsample_layer = ops.get_downsample_layer(self.hparams.downsample_layer)
        upsample_layer = ops.get_upsample_layer(self.hparams.upsample_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        conv_rnn_states = states['conv_rnn_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        image = tf.where(self.ground_truth[t], inputs['images'], states['gen_image'])  # schedule sampling (if any)
        last_images = states['last_images'][1:] + [image]
        if 'pix_distribs' in inputs:
            pix_distrib = tf.where(self.ground_truth[t], inputs['pix_distribs'], states['gen_pix_distrib'])
            last_pix_distribs = states['last_pix_distribs'][1:] + [pix_distrib]
        if 'states' in inputs:
            state = tf.where(self.ground_truth[t], inputs['states'], states['gen_state'])

        state_action = []
        state_action_z = []
        if 'encoded_actions' in inputs:
            state_action.append(inputs['encoded_actions'])
            state_action_z.append(inputs['encoded_actions'])
        elif 'actions' in inputs:
            state_action.append(inputs['actions'])
            state_action_z.append(inputs['actions'])

        if 'dx' in inputs:
            state_action_z.append(inputs['dx'])

        if 'states' in inputs:
            state_action.append(state)
            # don't backpropagate the convnet through the state dynamics
            state_action_z.append(tf.stop_gradient(state))

        if 'zs' in inputs:
            if self.hparams.use_rnn_z:
                with tf.variable_scope('%s_z' % self.hparams.rnn):
                    rnn_z, rnn_z_state = self._rnn_func(inputs['zs'], states['rnn_z_state'], self.hparams.nz)
                state_action_z.append(rnn_z)
            else:
                state_action_z.append(inputs['zs'])

        def concat(tensors, axis):
            if len(tensors) == 0:
                return tf.zeros([batch_size, 0])
            elif len(tensors) == 1:
                return tensors[0]
            else:
                return tf.concat(tensors, axis=axis)
        state_action = concat(state_action, axis=-1)
        state_action_z = concat(state_action_z, axis=-1)
        if 'encoded_actions' in inputs:
            gen_input = tile_concat([image, inputs['encoded_actions'][:, None, None, :]], axis=-1)
        elif 'actions' in inputs:
            gen_input = tile_concat([image, inputs['actions'][:, None, None, :]], axis=-1)
        else:
            gen_input = image

        layers = []
        new_conv_rnn_states = []
        for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    h = tf.concat([image, self.inputs['images'][0]], axis=-1)
                    kernel_size = (5, 5)
                else:
                    h = layers[-1][-1]
                    kernel_size = (3, 3)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'input' and i == 0):
                    h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                h = downsample_layer(h, out_channels, kernel_size=kernel_size, strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, i)):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        conv_rnn_h = h
                    conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))

        num_encoder_layers = len(layers)
        for i, (out_channels, use_conv_rnn) in enumerate(self.decoder_layer_specs):
            with tf.variable_scope('h%d' % len(layers)):
                if i == 0:
                    h = layers[-1][-1]
                else:
                    h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'middle' and i == 0):
                    h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                h = upsample_layer(h, out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, len(layers))):
                    if self.hparams.where_add == 'all':
                        conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        conv_rnn_h = h
                    conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))
        assert len(new_conv_rnn_states) == len(conv_rnn_states)

        if self.hparams.last_frames and self.hparams.num_transformed_images:
            if self.hparams.transformation == 'flow':
                with tf.variable_scope('h%d_flow' % len(layers)):
                    h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_flow = norm_layer(h_flow)
                    h_flow = tf.nn.relu(h_flow)

                with tf.variable_scope('flows'):
                    flows = conv2d(h_flow, 2 * self.hparams.last_frames * self.hparams.num_transformed_images, kernel_size=(3, 3), strides=(1, 1))
                    flows = tf.reshape(flows, [batch_size, height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            else:
                assert len(self.hparams.kernel_size) == 2
                kernel_shape = list(self.hparams.kernel_size) + [self.hparams.last_frames * self.hparams.num_transformed_images]
                if self.hparams.transformation == 'dna':
                    with tf.variable_scope('h%d_dna_kernel' % len(layers)):
                        h_dna_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_dna_kernel = norm_layer(h_dna_kernel)
                        h_dna_kernel = tf.nn.relu(h_dna_kernel)

                    # Using largest hidden state for predicting untied conv kernels.
                    with tf.variable_scope('dna_kernels'):
                        kernels = conv2d(h_dna_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                        kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, None, None, :, :, None]
                    kernel_spatial_axes = [3, 4]
                elif self.hparams.transformation == 'cdna':
                    with tf.variable_scope('cdna_kernels'):
                        smallest_layer = layers[num_encoder_layers - 1][-1]
                        kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                        kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, :, :, None]
                    kernel_spatial_axes = [1, 2]
                else:
                    raise ValueError('Invalid transformation %s' % self.hparams.transformation)

            if self.hparams.transformation != 'flow':
                with tf.name_scope('kernel_normalization'):
                    kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
                    kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keepdims=True)

        if self.hparams.generate_scratch_image:
            with tf.variable_scope('h%d_scratch' % len(layers)):
                h_scratch = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                h_scratch = norm_layer(h_scratch)
                h_scratch = tf.nn.relu(h_scratch)

            # Using largest hidden state for predicting a new image layer.
            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            with tf.variable_scope('scratch_image'):
                scratch_image = conv2d(h_scratch, color_channels, kernel_size=(3, 3), strides=(1, 1))
                scratch_image = tf.nn.sigmoid(scratch_image)

        with tf.name_scope('transformed_images'):
            transformed_images = []
            if self.hparams.last_frames and self.hparams.num_transformed_images:
                if self.hparams.transformation == 'flow':
                    transformed_images.extend(apply_flows(last_images, flows))
                else:
                    transformed_images.extend(apply_kernels(last_images, kernels, self.hparams.dilation_rate))
            if self.hparams.prev_image_background:
                transformed_images.append(image)
            if self.hparams.first_image_background and not self.hparams.context_images_background:
                transformed_images.append(self.inputs['images'][0])
            if self.hparams.context_images_background:
                transformed_images.extend(tf.unstack(self.inputs['images'][:self.hparams.context_frames]))
            if self.hparams.generate_scratch_image:
                transformed_images.append(scratch_image)

        if 'pix_distribs' in inputs:
            with tf.name_scope('transformed_pix_distribs'):
                transformed_pix_distribs = []
                if self.hparams.last_frames and self.hparams.num_transformed_images:
                    if self.hparams.transformation == 'flow':
                        transformed_pix_distribs.extend(apply_flows(last_pix_distribs, flows))
                    else:
                        transformed_pix_distribs.extend(apply_kernels(last_pix_distribs, kernels, self.hparams.dilation_rate))
                if self.hparams.prev_image_background:
                    transformed_pix_distribs.append(pix_distrib)
                if self.hparams.first_image_background and not self.hparams.context_images_background:
                    transformed_pix_distribs.append(self.inputs['pix_distribs'][0])
                if self.hparams.context_images_background:
                    transformed_pix_distribs.extend(tf.unstack(self.inputs['pix_distribs'][:self.hparams.context_frames]))
                if self.hparams.generate_scratch_image:
                    transformed_pix_distribs.append(pix_distrib)

        with tf.name_scope('masks'):
            if len(transformed_images) > 1:
                with tf.variable_scope('h%d_masks' % len(layers)):
                    h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_masks = norm_layer(h_masks)
                    h_masks = tf.nn.relu(h_masks)

                with tf.variable_scope('masks'):
                    if self.hparams.dependent_mask:
                        h_masks = tf.concat([h_masks] + transformed_images, axis=-1)
                    masks = conv2d(h_masks, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
                    masks = tf.nn.softmax(masks)
                    masks = tf.split(masks, len(transformed_images), axis=-1)
            elif len(transformed_images) == 1:
                masks = [tf.ones([batch_size, height, width, 1])]
            else:
                raise ValueError("Either one of the following should be true: "
                                 "last_frames and num_transformed_images, first_image_background, "
                                 "prev_image_background, generate_scratch_image")

        with tf.name_scope('gen_images'):
            assert len(transformed_images) == len(masks)
            gen_image = tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])

        if 'pix_distribs' in inputs:
            with tf.name_scope('gen_pix_distribs'):
                assert len(transformed_pix_distribs) == len(masks)
                gen_pix_distrib = tf.add_n([transformed_pix_distrib * mask
                                            for transformed_pix_distrib, mask in zip(transformed_pix_distribs, masks)])
                if self.hparams.renormalize_pixdistrib:
                    gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keepdims=True)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'gen_inputs': gen_input,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1)}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distribs'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state
        if self.hparams.transformation == 'flow':
            outputs['gen_flows'] = flows
            flows_transposed = tf.transpose(flows, [0, 1, 2, 4, 3])
            flows_rgb_transposed = tf_utils.flow_to_rgb(flows_transposed)
            flows_rgb = tf.transpose(flows_rgb_transposed, [0, 1, 2, 4, 3])
            outputs['gen_flows_rgb'] = flows_rgb

        new_states = {'time': time + 1,
                      'gen_image': gen_image,
                      'last_images': last_images,
                      'conv_rnn_states': new_conv_rnn_states}
        if 'zs' in inputs and self.hparams.use_rnn_z:
            new_states['rnn_z_state'] = rnn_z_state
        if 'pix_distribs' in inputs:
            new_states['gen_pix_distrib'] = gen_pix_distrib
            new_states['last_pix_distribs'] = last_pix_distribs
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states


def generator_fn(inputs, mode, outputs_enc=None, hparams=None):
    

    inputs['actions'] = inputs['actions'] / tf.constant([0.07, 0.07, 0.5, 0.15])

    # if hparams is not None:
        # print("generator hparams:", hparams)

    batch_size = inputs['images'].shape[1].value


    if hparams.use_encoded_actions:
        action_probs = action_encoder_fn(inputs, hparams=hparams)
        if mode == 'test':
            eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.encoded_action_size], 0, 1)
        else:
            eps = tf.random_normal([hparams.sequence_length - 1, 9, hparams.encoded_action_size], 0, 1)
        inputs['encoded_actions'] = action_probs['action_mu'] + tf.exp(action_probs['action_log_sigma_sq'] / 2.0) * eps

#        use_actions = tf.reshape(inputs['use_action'], [-1, 1])
#        use_actions = tf.tile(use_actions, [1, hparams.encoded_action_size])
#        use_actions = tf.reshape(use_actions, [inputs['encoded_actions'].shape[0], inputs['encoded_actions'].shape[1], inputs['encoded_actions'].shape[2]])
#        inputs['use_actions_array_encoded'] = use_actions

    if hparams.train_with_partial_actions:
        assert hparams.use_encoded_actions or hparams.deterministic_inverse, "Training without actions requires using encoded actions"

        if hparams.learn_z_seq_prior:
            # print("Learn z_seq_priot\n\n\n\n\n")
            # Learn prior for robot data
            r_prior_z_mu = tf.get_variable('r_prior_z_mu', initializer=tf.zeros([hparams.encoded_action_size]))
            r_prior_z_log_sigma_sq = tf.get_variable('r_prior_z_log_sigma_sq', initializer=tf.zeros([hparams.encoded_action_size]))

            with tf.variable_scope('image_encoder'):
                enc_image_probs = image_encoder_fn(inputs, hparams=hparams)

            eps = tf.random_normal([batch_size, 2*hparams.encoded_action_size], 0, 1)
            enc_image = enc_image_probs['enc_image_mu'] + \
                tf.exp(enc_image_probs['enc_image_log_sigma_sq'] / 2.0) * eps

            # Learn prior for human data
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=2 * hparams.encoded_action_size)
            with tf.variable_scope('z_lstm') as lstm_scope:
                zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                _, state = lstm_cell(enc_image, zero_state)

                lstm_scope.reuse_variables()
                lstm_outputs = []
                output = tf.zeros([batch_size, 2 * hparams.encoded_action_size])
                for t in range(hparams.sequence_length - 1):
                    output, state = lstm_cell(output, state)
                    lstm_outputs.append(output)
            lstm_outputs = tf.convert_to_tensor(lstm_outputs)
            h_prior_z_mu = lstm_outputs[:, :, :hparams.encoded_action_size]
            h_prior_z_log_sigma_sq = lstm_outputs[:, :, hparams.encoded_action_size:]

        elif hparams.nda:
            # print("NDA,\n\n\n\n\n\n\n\n\n")
            repeats = [9, 3]
            if hparams.deterministic_da:
                da = [tf.get_variable('da0', initializer=tf.zeros([hparams.nda])),
                      tf.get_variable('da1', initializer=tf.zeros([hparams.nda]))]
                if mode != 'train':
                    tiled_da = tf.tile(tf.reshape(da[hparams.val_action_domain], [1, 1, hparams.nda]), [hparams.sequence_length - 1, batch_size, 1])
                    inputs['da'] = tiled_da
                else:
                    tiled_da = [tf.tile(tf.reshape(m, [1, 1, hparams.nda]), [hparams.sequence_length - 1, r, 1]) for m, r in zip(da, repeats)]
                    concat_da = tf.concat(tiled_da, axis=1)
                    inputs['da'] = concat_da
            else:
                da_mu = [tf.get_variable('da0_mu', initializer=tf.zeros([hparams.nda])),
                         tf.get_variable('da1_mu', initializer=tf.zeros([hparams.nda]))]
                da_log_sigma_sq = [tf.get_variable('da0_log_sigma_sq', initializer=tf.zeros([hparams.nda])),
                                tf.get_variable('da1_log_sigma_sq', initializer=tf.zeros([hparams.nda]))]

                if mode != 'train':
                    tiled_da_mu = tf.tile(tf.reshape(da_mu[hparams.val_action_domain], [1, 1, hparams.nda]), [hparams.sequence_length - 1, batch_size , 1])
                    tiled_da_log_sigma_sq = tf.tile(tf.reshape(da_log_sigma_sq[hparams.val_action_domain], [1, 1, hparams.nda]), [hparams.sequence_length - 1, batch_size, 1])
                    eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nda], 0, 1)
                    da = tiled_da_mu + tf.exp(tiled_da_log_sigma_sq / 2.0) * eps
                    inputs['da'] = da
                else:
                    tiled_da_mu = [tf.tile(tf.reshape(m, [1, 1, hparams.nda]), [hparams.sequence_length - 1, r, 1]) for m, r in zip(da_mu, repeats)]
                    tiled_da_log_sigma_sq = [tf.tile(tf.reshape(s, [1, 1, hparams.nda]), [hparams.sequence_length - 1, r, 1]) for s, r in zip(da_log_sigma_sq, repeats)]

                    concat_da_mu = tf.concat(tiled_da_mu, axis=1)
                    concat_da_log_sigma_sq = tf.concat(tiled_da_log_sigma_sq, axis=1)

                    eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nda], 0, 1)
                    da = concat_da_mu + tf.exp(concat_da_log_sigma_sq / 2.0) * eps
                    inputs['da'] = da

        if mode != 'test':
            with tf.variable_scope('inverse_model'):
                inverse_action_probs = inverse_model_fn(inputs, hparams=hparams)

            if not hparams.deterministic_inverse:
                eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.encoded_action_size], 0, 1)
                additional_encoded_actions = {}
                additional_encoded_actions['encoded_actions_inverse'] = inverse_action_probs['action_inverse_mu'] + \
                    tf.exp(inverse_action_probs['action_inverse_log_sigma_sq'] / 2.0) * eps
                additional_encoded_actions['original_encoded_actions'] = inputs['encoded_actions'] - 0

    #            inputs['encoded_actions'] = tf.where(use_actions, x=inputs['encoded_actions'], y=inputs['encoded_actions_inverse'])
                inputs['encoded_actions'] = tf.concat([inputs['encoded_actions'], additional_encoded_actions['encoded_actions_inverse'][:, 9:, :]], axis=1)
                # print("Encoded actions.shape", inputs['encoded_actions'].shape)
            else:
                assert not hparams.use_encoded_actions
                inputs['actions_inverse'] = inverse_action_probs['action_inverse_mu']

                use_actions = tf.reshape(inputs['use_action'], [-1, 1])
                use_actions = tf.tile(use_actions, [1, inputs['actions'].shape[-1]])
                use_actions = tf.reshape(use_actions, [inputs['actions'].shape[0], inputs['actions'].shape[1], inputs['actions'].shape[2]])
                inputs['encoded_actions'] = tf.where(use_actions, x=inputs['actions'], y=inputs['actions_inverse'])

        if hparams.ndx:
            # print("NDX,\n\n\n\n\n\n\n\n\n")
            repeats = [9, 3]
            if hparams.deterministic_dx:
                dx = [tf.get_variable('dx0', initializer=tf.zeros([hparams.ndx])),
                      tf.get_variable('dx1', initializer=tf.zeros([hparams.ndx]))]
                if mode != 'train':
                    tiled_dx = tf.tile(tf.reshape(dx[hparams.val_visual_domain], [1, 1, hparams.ndx]), [hparams.sequence_length - 1, batch_size, 1])
                    inputs['dx'] = tiled_dx
                else:
                    tiled_dx = [tf.tile(tf.reshape(m, [1, 1, hparams.ndx]), [hparams.sequence_length - 1, r, 1]) for m, r in zip(dx, repeats)]
                    concat_dx = tf.concat(tiled_dx, axis=1)
                    inputs['dx'] = concat_dx
            else:
                dx_mu = [tf.get_variable('dx0_mu', initializer=tf.zeros([hparams.ndx])),
                         tf.get_variable('dx1_mu', initializer=tf.zeros([hparams.ndx]))]
                dx_log_sigma_sq = [tf.get_variable('dx0_log_sigma_sq', initializer=tf.zeros([hparams.ndx])),
                                tf.get_variable('dx1_log_sigma_sq', initializer=tf.zeros([hparams.ndx]))]

                if mode != 'train':
                    tiled_dx_mu = tf.tile(tf.reshape(dx_mu[hparams.val_visual_domain], [1, 1, hparams.ndx]), [hparams.sequence_length - 1, batch_size , 1])
                    tiled_dx_log_sigma_sq = tf.tile(tf.reshape(dx_log_sigma_sq[hparams.val_visual_domain], [1, 1, hparams.ndx]), [hparams.sequence_length - 1, batch_size, 1])
                    eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.ndx], 0, 1)
                    dx = tiled_dx_mu + tf.exp(tiled_dx_log_sigma_sq / 2.0) * eps
                    inputs['dx'] = dx
                else:
                    tiled_dx_mu = [tf.tile(tf.reshape(m, [1, 1, hparams.ndx]), [hparams.sequence_length - 1, r, 1]) for m, r in zip(dx_mu, repeats)]
                    tiled_dx_log_sigma_sq = [tf.tile(tf.reshape(s, [1, 1, hparams.ndx]), [hparams.sequence_length - 1, r, 1]) for s, r in zip(dx_log_sigma_sq, repeats)]

                    concat_dx_mu = tf.concat(tiled_dx_mu, axis=1)
                    concat_dx_log_sigma_sq = tf.concat(tiled_dx_log_sigma_sq, axis=1)

                    eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.ndx], 0, 1)
                    dx = concat_dx_mu + tf.exp(concat_dx_log_sigma_sq / 2.0) * eps
                    inputs['dx'] = dx


    if mode != 'test':
        if hparams.decode_actions and hparams.use_encoded_actions:
            decoded_actions = {}
            decoded_actions['decoded_actions'] = action_decoder_fn(inputs['encoded_actions'], inputs['actions'].shape, hparams=hparams)
    #        decoded_actions['decoded_actions'] = action_probs['action_mu']
            if hparams.decode_from_inverse:
                # print("Decoding from inverse:")
                decoded_actions['decoded_actions_inverse'] = action_decoder_fn(additional_encoded_actions['encoded_actions_inverse'], inputs['actions'].shape, hparams=hparams)

    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}
    if hparams.nz:
        def sample_zs():
            if outputs_enc is None:
                zs = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
            else:
                enc_zs_mu = outputs_enc['enc_zs_mu']
                enc_zs_log_sigma_sq = outputs_enc['enc_zs_log_sigma_sq']
                eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
                zs = enc_zs_mu + tf.exp(enc_zs_log_sigma_sq / 2.0) * eps
            return zs
        inputs['zs'] = sample_zs()
    else:
        if outputs_enc is not None:
            raise ValueError('outputs_enc has to be None when nz is 0.')
    if not hparams.train_with_partial_actions:
        hparams.batch_size = 9
        inputs['images'] = inputs['images'][:, :9, :, :, :]
        inputs['use_action'] = inputs['use_action'][:, :9, :]
        inputs['actions'] = inputs['actions'][:, :9, :]

    elif mode == 'train' and hparams.predict_from_inverse:
        hparams.batch_size = 21
        inputs['images'] = tf.concat([inputs['images'], inputs['images'][:, :9, :, :, :]], axis=1)
        inputs['use_action'] = tf.concat([inputs['use_action'], inputs['use_action'][:, :9, :]], axis=1)
        inputs['actions'] = tf.concat([inputs['actions'], inputs['actions'][:, :9, :]], axis=1)
        inputs['encoded_actions'] = tf.concat([inputs['encoded_actions'], additional_encoded_actions['encoded_actions_inverse'][:, :9, :]], axis=1)

        if 'da' in inputs.keys():
            inputs['da'] = tf.concat([inputs['da'], inputs['da'][:, :9, :]], axis=1)
        if 'dx' in inputs.keys():
            inputs['dx'] = tf.concat([inputs['dx'], inputs['dx'][:, :9, :]], axis=1)


    cell = DNACell(inputs, hparams)

    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                   swap_memory=False, time_major=True)

    if hparams.nz:
        inputs_samples = {name: flatten(tf.tile(input[:, None], [1, hparams.num_samples] + [1] * (input.shape.ndims - 1)), 1, 2)
                          for name, input in inputs.items() if name != 'zs'}
        inputs_samples['zs'] = tf.concat([sample_zs() for _ in range(hparams.num_samples)], axis=1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            cell_samples = DNACell(inputs_samples, hparams)
            outputs_samples, _ = tf.nn.dynamic_rnn(cell_samples, inputs_samples, dtype=tf.float32,
                                                   swap_memory=False, time_major=True)
        gen_images_samples = outputs_samples['gen_images']
        gen_images_samples = tf.stack(tf.split(gen_images_samples, hparams.num_samples, axis=1), axis=-1)
        gen_images_samples_avg = tf.reduce_mean(gen_images_samples, axis=-1)
        outputs['gen_images_samples'] = gen_images_samples
        outputs['gen_images_samples_avg'] = gen_images_samples_avg
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
    gen_images = outputs['gen_images']
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
    if mode != 'test':
        if hparams.use_encoded_actions:
            for k in action_probs:
                outputs[k] = action_probs[k]
            if hparams.decode_actions:
                for k in decoded_actions:
                    outputs[k] = decoded_actions[k]
                
        if hparams.train_with_partial_actions:
            for k in inverse_action_probs:
                outputs[k] = inverse_action_probs[k]
            for k in additional_encoded_actions:
                outputs[k] = additional_encoded_actions[k]

            if hparams.learn_z_seq_prior:
                outputs['r_prior_z_mu'] = r_prior_z_mu
                outputs['r_prior_z_log_sigma_sq'] = r_prior_z_log_sigma_sq
                outputs['h_prior_z_mu'] = h_prior_z_mu
                outputs['h_prior_z_log_sigma_sq'] = h_prior_z_log_sigma_sq

            elif hparams.nda and not hparams.deterministic_da:
                outputs['da0_mu'] = da_mu[0]
                outputs['da0_log_sigma_sq'] = da_log_sigma_sq[0]
                outputs['da1_mu'] = da_mu[1]
                outputs['da1_log_sigma_sq'] = da_log_sigma_sq[1]

            if hparams.ndx and not hparams.deterministic_dx:
                outputs['dx0_mu'] = dx_mu[0]
                outputs['dx0_log_sigma_sq'] = dx_log_sigma_sq[0]
                outputs['dx1_mu'] = dx_mu[1]
                outputs['dx1_log_sigma_sq'] = dx_log_sigma_sq[1]

    return gen_images, outputs


class SAVPVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(SAVPVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, encoder_fn, *args, **kwargs)
        if self.hparams.e_net == 'none' or self.hparams.nz == 0:
            self.encoder_fn = None
        else:
            assert self.hparams.e_net != 'nlayers_context'

        if self.hparams.d_net == 'none':
            self.discriminator_fn = None
        self.deterministic = not self.hparams.nz
        # print(self.hparams.context_frames)

    def get_default_hparams_dict(self):
        default_hparams = super(SAVPVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=1.0,
            l2_weight=0.0,
            d_net='legacy',
            n_layers=3,
            ndf=32,
            norm_layer='instance',
            d_downsample_layer='conv_pool2d',
            d_conditional=True,
            d_use_gt_inputs=True,
            ngf=32,
            downsample_layer='conv_pool2d',
            upsample_layer='upsample_conv2d',
            transformation='cdna',
            kernel_size=(5, 5),
            dilation_rate=(1, 1),
            where_add='all',
            rnn='lstm',
            conv_rnn='lstm',
            num_transformed_images=4,
            last_frames=1,
            prev_image_background=True,
            first_image_background=True,
            context_images_background=False,
            generate_scratch_image=True,
            dependent_mask=True,
            schedule_sampling='inverse_sigmoid',
            schedule_sampling_k=900.0,
            schedule_sampling_steps=(0, 100000),
            e_net='n_layer',
            use_e_rnn=False,
            inverse_model_net='n_layer',
            use_inverse_model_rnn=False,
            nz=8,
            num_samples=8,
            nef=64,
            use_rnn_z=True,
            ablation_conv_rnn_norm=False,
            renormalize_pixdistrib=True,
            use_encoded_actions=False,
            decode_actions=False,
            action_decoder_mse_weight=1.0,
            action_encoder_channels=16,
            action_encoder_layers=3,
            action_encoder_norm_layer='none',
            encoded_action_size=4,
            action_encoder_kl_weight=0.1,
            inverse_action_encoder_kl_weight=0.1,
            train_with_partial_actions=False,
            action_js_loss=0.1,
            deterministic_inverse=False,
            deterministic_inverse_mse=1.0,
            decode_from_inverse=False,
            use_domain_adaptation=False,
            ndx=0,#16,
            deterministic_dx=False,
            dx_kl_weight=0.001,
            val_visual_domain=0,    # 0 for robot, 1 for human
            nda=0,#8,
            deterministic_da=False,
            da_kl_weight=0.001,
            val_action_domain=1,    # 0 for robot, 1 for human
            learn_z_seq_prior=False,
            kl_on_inverse=False,
            action_inverse_kl_weight=-1.0,
            predict_from_inverse=False,
            add_dx_to_inverse=True,
            add_da_to_inverse=True
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parse_hparams(self, hparams_dict, hparams):
        hparams = super(SAVPVideoPredictionModel, self).parse_hparams(hparams_dict, hparams)
        if self.mode == 'test':
            def override_hparams_maybe(name, value):
                orig_value = hparams.values()[name]
                if orig_value != value:
                    print('Overriding hparams from %s=%r to %r for mode=%s.' %
                          (name, orig_value, value, self.mode))
                    hparams.set_hparam(name, value)
            override_hparams_maybe('d_net', 'none')
            override_hparams_maybe('e_net', 'none')
            override_hparams_maybe('schedule_sampling', 'none')
        return hparams
    
    def generator_loss_fn(self, inputs, outputs, targets):
        if not self.hparams.train_with_partial_actions:
            targets = targets[:, :9, :, :, :]
#        elif self.mode == 'train' and self.hparams.predict_from_inverse:
#            targets = tf.concat([targets, targets[:, :9, :, :, :]], axis=1)
        gen_losses = super(SAVPVideoPredictionModel, self).generator_loss_fn(inputs, outputs, targets)

        #r_idx = tf.where(tf.reshape(inputs['use_actions_array_encoded'], [-1]))
        #h_idx = tf.where(tf.reshape(~inputs['use_actions_array_encoded'], [-1]))
        if self.hparams.use_encoded_actions:

            tmp_mu = outputs['action_mu'][:, :9, :]
            tmp_log_sigma_sq = outputs['action_log_sigma_sq'][:, :9, :]
            
            if not self.hparams.learn_z_seq_prior:
                action_encoder_kl_loss = kl_loss(tmp_mu, tmp_log_sigma_sq)
            else:
                action_encoder_kl_loss = kl_loss_dist(tmp_mu, tmp_log_sigma_sq, outputs['r_prior_z_mu'], outputs['r_prior_z_log_sigma_sq'])

            gen_losses['action_encoder_kl_loss'] = (action_encoder_kl_loss, self.hparams.action_encoder_kl_weight)

            if self.hparams.decode_actions:
                tmp_decoded_actions = outputs['decoded_actions'][:, :9, :]
                tmp_gt_actions = inputs['actions'][:, :9, :]

                # tmp_gt_actions = tf.Print(tmp_gt_actions, [tmp_gt_actions], "Gt actions")

                decoded_action_loss = l2_loss(tmp_decoded_actions, tmp_gt_actions)
                # decoded_action_loss = tf.Print(decoded_action_loss, [decoded_action_loss], "decoded action_loss")   
                gen_losses['action_decoder_mse'] = (decoded_action_loss, self.hparams.action_decoder_mse_weight)



        if self.hparams.train_with_partial_actions and not self.hparams.deterministic_inverse:

            if self.hparams.nda and self.hparams.da_kl_weight and not self.hparams.deterministic_da:
                r_da_kl_loss = kl_loss(outputs['da0_mu'], outputs['da0_log_sigma_sq'])
                h_da_kl_loss = kl_loss(outputs['da1_mu'], outputs['da1_log_sigma_sq'])

                da_kl_loss = r_da_kl_loss + h_da_kl_loss
                gen_losses['da_kl_loss'] = (da_kl_loss, self.hparams.da_kl_weight)

            if self.hparams.ndx and self.hparams.dx_kl_weight and not self.hparams.deterministic_dx:
                r_dx_kl_loss = kl_loss(outputs['dx0_mu'], outputs['dx0_log_sigma_sq'])
                h_dx_kl_loss = kl_loss(outputs['dx1_mu'], outputs['dx1_log_sigma_sq'])

                dx_kl_loss = r_dx_kl_loss + h_dx_kl_loss
                gen_losses['dx_kl_loss'] = (dx_kl_loss, self.hparams.dx_kl_weight)


            if not self.hparams.learn_z_seq_prior:
                inverse_action_encoder_kl_loss = kl_loss(outputs['action_inverse_mu'], outputs['action_inverse_log_sigma_sq'])
            else:
                tmp_inverse_mu = outputs['action_inverse_mu'][:, :9, :]
                tmp_inverse_log_sigma_sq = outputs['action_inverse_log_sigma_sq'][:, :9, :]

                r_kl_loss = kl_loss_dist(tmp_inverse_mu, tmp_inverse_log_sigma_sq, outputs['r_prior_z_mu'], outputs['r_prior_z_log_sigma_sq'])

                if self.mode == 'train':
                    tmp_mu_1 = outputs['action_inverse_mu'][:, 9:, :]
                    tmp_log_sigma_sq_1 = outputs['action_inverse_log_sigma_sq'][:, 9:, :]
                    #tmp_mu_1 = tf.gather(outputs['action_inverse_mu'], h_idx, axis=1)
                    #tmp_log_sigma_sq_1 = tf.gather(outputs['action_inverse_log_sigma_sq'], h_idx, axis=1)

                    tmp_mu_2 = outputs['h_prior_z_mu'][:, 9:, :]
                    tmp_log_sigma_sq_2 = outputs['h_prior_z_log_sigma_sq'][:, 9:, :]
                    #tmp_mu_2 = tf.gather(outputs['h_prior_z_mu'], h_idx, axis=1)
                    #tmp_log_sigma_sq_2 = tf.gather(outputs['h_prior_z_log_sigma_sq'], h_idx, axis=1)

                    h_kl_loss = kl_loss_dist(tmp_mu_1, tmp_log_sigma_sq_1, tmp_mu_2, tmp_log_sigma_sq_2)
                    inverse_action_encoder_kl_loss = r_kl_loss + h_kl_loss
                else:
                    inverse_action_encoder_kl_loss = r_kl_loss

            gen_losses['action_inverse_encoder_kl_loss'] = (inverse_action_encoder_kl_loss, self.hparams.inverse_action_encoder_kl_weight)

            tmp_inverse_mu = outputs['action_inverse_mu'][:, :9, :]
            tmp_inverse_log_sigma_sq = outputs['action_inverse_log_sigma_sq'][:, :9, :]
            
#            tmp_inverse_mu = tf.gather(tf.reshape(outputs['action_inverse_mu'], [-1]), idx)
#            tmp_inverse_mu = tf.reshape(tmp_inverse_mu, [outputs['action_inverse_mu'].shape[0], -1, outputs['action_inverse_mu'].shape[-1]])
#            tmp_inverse_log_sigma_sq = tf.gather(tf.reshape(outputs['action_inverse_log_sigma_sq'], [-1]), idx)
#            tmp_inverse_log_sigma_sq = tf.reshape(tmp_inverse_log_sigma_sq, [outputs['action_inverse_log_sigma_sq'].shape[0], -1, outputs['action_inverse_log_sigma_sq'].shape[-1]])

            action_js_loss = js_loss(tmp_mu, tmp_log_sigma_sq, tmp_inverse_mu, tmp_inverse_log_sigma_sq)
#            action_js_loss = js_loss(outputs['action_mu'], outputs['action_log_sigma_sq'],
#                                     outputs['action_inverse_mu'], outputs['action_inverse_log_sigma_sq'])

            gen_losses['action_js_loss'] = (action_js_loss, self.hparams.action_js_loss)

            if self.hparams.decode_actions and self.hparams.decode_from_inverse:
                tmp_decoded_inverse = outputs['decoded_actions_inverse'][:, :9, :]
                tmp_gt_actions = inputs['actions'][:, :9, :]
                decoded_inverse_action_loss = l2_loss(tmp_decoded_inverse, tmp_gt_actions)
                gen_losses['action_decoder_inverse_mse'] = (decoded_inverse_action_loss, self.hparams.action_decoder_mse_weight)

            if self.hparams.kl_on_inverse:
                action_inverse_kl_loss = kl_loss(tmp_inverse_mu, tmp_inverse_log_sigma_sq)
                weight = self.hparams.action_inverse_kl_weight
                if weight < 0:
                    weight = self.hparams.action_encoder_kl_weight
                gen_losses['action_inverse_kl_loss'] = (action_inverse_kl_loss, weight)

        elif self.hparams.train_with_partial_actions:
            inverse_mse = l2_loss(inputs['actions_inverse'], inputs['encoded_actions'])
            gen_losses['action_deterministic_inverse_mse'] = (inverse_mse, self.hparams.deterministic_inverse_mse)

        return gen_losses


def apply_dna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 6-D of shape
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, height, width, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]

    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, height, width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Combine channel and batch dimensions into the first dimension.
    image_transposed = tf.transpose(image_padded, [3, 0, 1, 2])
    image_reshaped = flatten(image_transposed, 0, 1)[..., None]
    patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                strides=[1] * 4, rates=[1] + dilation_rate + [1], padding='VALID')
    # Separate channel and batch dimensions, and move channel dimension.
    patches_transposed = tf.reshape(patches_reshaped, [color_channels, batch_size, height, width, kernel_size[0] * kernel_size[1]])
    patches = tf.transpose(patches_transposed, [1, 2, 3, 0, 4])
    # Reduce along the spatial dimensions of the kernel.
    outputs = tf.matmul(patches, kernels_reshaped)
    outputs = tf.unstack(outputs, axis=-1)
    return outputs


def apply_cdna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def apply_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    if isinstance(image, list):
        image_list = image
        kernels_list = tf.split(kernels, len(image_list), axis=-1)
        outputs = []
        for image, kernels in zip(image_list, kernels_list):
            outputs.extend(apply_kernels(image, kernels))
    else:
        if len(kernels.get_shape()) == 4:
            outputs = apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
        elif len(kernels.get_shape()) == 6:
            outputs = apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)
        else:
            raise ValueError
    return outputs


def apply_flows(image, flows):
    if isinstance(image, list):
        image_list = image
        flows_list = tf.split(flows, len(image_list), axis=-1)
        outputs = []
        for image, flows in zip(image_list, flows_list):
            outputs.extend(apply_flows(image, flows))
    else:
        flows = tf.unstack(flows, axis=-1)
        outputs = [flow_ops.image_warp(image, flow) for flow in flows]
    return outputs


def identity_kernel(kernel_size):
    kh, kw = kernel_size
    kernel = np.zeros(kernel_size)

    def center_slice(k):
        if k % 2 == 0:
            return slice(k // 2 - 1, k // 2 + 1)
        else:
            return slice(k // 2, k // 2 + 1)

    kernel[center_slice(kh), center_slice(kw)] = 1.0
    kernel /= np.sum(kernel)
    return kernel
