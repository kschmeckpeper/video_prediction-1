import tensorflow as tf

from video_prediction.ops import sigmoid_kl_with_logits


def l1_loss(pred, target):
    return tf.reduce_mean(tf.abs(target - pred))


def l2_loss(pred, target):
    return tf.reduce_mean(tf.square(target - pred))


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = tf.reduce_mean(tf.square(logits - labels))
    elif gan_loss_type == 'SNGAN':
        # this is the form of the loss used in the official implementation of the SNGAN paper, but it leads to
        # worse results in our video prediction experiments
        if labels == 0.0:
            loss = tf.reduce_mean(tf.nn.softplus(logits))
        elif labels == 1.0:
            loss = tf.reduce_mean(tf.nn.softplus(-logits))
        else:
            raise NotImplementedError
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def kl_loss(mu, log_sigma_sq):
    sigma_sq = tf.exp(log_sigma_sq)
    return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - sigma_sq, axis=-1))

def kl_loss_dist(mu_1, log_sigma_sq_1, mu_2, log_sigma_sq_2):
    sigma_sq_1 = tf.exp(log_sigma_sq_1)
    sigma_sq_2 = tf.exp(log_sigma_sq_2)
    sigma_1 = tf.sqrt(sigma_sq_1)
    sigma_2 = tf.sqrt(sigma_sq_2)
    
    kl = tf.log(sigma_2 / sigma_1) + (sigma_sq_1 + (mu_1 - mu_2)**2) / (2 * sigma_sq_2) - 0.5
    print("kl:", kl.shape)
    return tf.reduce_mean(kl)

def js_loss(mu_1, log_sigma_sq_1, mu_2, log_sigma_sq_2):
    # This is an aproximation of the JS divergence that approximates the mixture of
    # two gaussians as a single gaussian instead of a multi-modal distribution
    mean_mu = (mu_1 + mu_2) / 2
    mean_log_sigma_sq = tf.log((tf.exp(log_sigma_sq_1) + tf.exp(log_sigma_sq_2)) / 2)

    return kl_loss_dist(mu_1, log_sigma_sq_1, mean_mu, mean_log_sigma_sq) + \
           kl_loss_dist(mu_2, log_sigma_sq_2, mean_mu, mean_log_sigma_sq)
    
def jeffreys_divergence(mu_1, log_sigma_sq_1, mu_2, log_sigma_sq_2):
    return kl_loss_dist(mu_1, log_sigma_sq_1, mu_2, log_sigma_sq_2) + \
           kl_loss_dist(mu_2, log_sigma_sq_2, mu_1, log_sigma_sq_1)


