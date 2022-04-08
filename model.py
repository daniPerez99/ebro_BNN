#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers

# MODEL FUNCTION
# =============================================================================
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

def create_model(dataset_size, num_features, l1_n, l2_n):
    input = tf.keras.Input(shape=(num_features,), name="input")
    layer1 = tfp.layers.DenseVariational(
            units=l1_n,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / dataset_size,
            activation="relu",
            name="variatonal_tfp_1")
    layer2 = tfp.layers.DenseVariational(
            units=l2_n,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / dataset_size,
            activation="relu",
            name="variatonal_tfp_2")
    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.   
    probabilistic_output = tfp.layers.IndependentNormal(1)
    output = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(2))
    model = tf.keras.Sequential([input,layer1,layer2,probabilistic_output,output])
   
    return model
