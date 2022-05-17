# -*- coding: utf-8 -*-
"""
Todo description
"""

import lib

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np

# strategy = tf.distribute.get_strategy()

# print('Number of replicas:', strategy.num_replicas_in_sync)

# AUTOTUNE = tf.data.experimental.AUTOTUNE
    
print(tf.__version__)

#%% INPUT IMAGES

GCS_PATH = 'data'

# this is the source domain
PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

# this is the target domain
MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
print('Monet TFRecord Files:', len(MONET_FILENAMES))


#%% IMAGE READING 


monet_ds = lib.load_dataset(MONET_FILENAMES, labeled=True).batch(1)
photo_ds = lib.load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)


#%% EXAMPLES

example_monet = next(iter(monet_ds))
example_photo = next(iter(photo_ds))

plt.subplot(121)
plt.title('Photo')
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet')
plt.imshow(example_monet[0] * 0.5 + 0.5)


#%% DISCRIMINATOR

# with strategy.scope():
monet_generator = lib.Generator() # transforms photos to Monet-esque paintings
photo_generator = lib.Generator() # transforms Monet paintings to be more like photos

monet_discriminator = lib.Discriminator() # differentiates real Monet paintings and generated Monet paintings
photo_discriminator = lib.Discriminator() # differentiates real photos and generated photos
    

to_monet = monet_generator(example_photo)

plt.subplot(1, 2, 1)
plt.title("Original Photo")
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(1, 2, 2)
plt.title("Monet-esque Photo")
plt.imshow(to_monet[0] * 0.5 + 0.5)
plt.show()

    
#%% LOSS FUNCTIONS
    
    
#%% TRAIN CYCLEGAN

# with strategy.scope():
monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    

# with strategy.scope():
cycle_gan_model = lib.CycleGan(
    monet_generator, photo_generator, monet_discriminator, photo_discriminator
)

cycle_gan_model.compile(
    m_gen_optimizer = monet_generator_optimizer,
    p_gen_optimizer = photo_generator_optimizer,
    m_disc_optimizer = monet_discriminator_optimizer,
    p_disc_optimizer = photo_discriminator_optimizer,
    gen_loss_fn = lib.generator_loss,
    disc_loss_fn = lib.discriminator_loss,
    cycle_loss_fn = lib.calc_cycle_loss,
    identity_loss_fn = lib.identity_loss
)

cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=25
)


#%% OUTPUT

_, ax = plt.subplots(5, 2, figsize=(12, 12))
for i, img in enumerate(photo_ds.take(5)):
    prediction = monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()


#%% SUBMISSION FILE FOR KAGGLE

i = 1
for img in photo_ds:
    prediction = lib.monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    im = lib.PIL.Image.fromarray(prediction)
    im.save("../images/" + str(i) + ".jpg")
    i += 1

import shutil
shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")