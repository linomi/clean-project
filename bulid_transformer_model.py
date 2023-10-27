from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import gc
from tensorflow import keras
from keras import layers
from keras.models import Model

class Tublet_projection(keras.layers.Layer):
    def __init__(self,patch_size,embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
                                        filters=embed_dim,
                                        kernel_size=patch_size,
                                        strides=patch_size,
                                        padding="VALID")
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))
    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def bulid_model(num_heads,num_layers,delay,embed_dim,output_shape):
    delay = delay
    num_heads =num_heads
    transformer_layers = num_layers
    embed_dim = embed_dim
    input_shape = (delay,304,608,1)
    LAYER_NORM_EPS = 1e-6
    output_shape = output_shape
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Tublet_projection(patch_size=(delay,10,10),embed_dim=embed_dim)(inputs)
    pathces = Lambda(lambda x : x/255.0)(patches)
    # Encode patches.
    encoded_patches = PositionalEncoder(embed_dim=embed_dim)(patches)
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    outputs = layers.Dense(units=output_shape, activation="linear")(representation)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model