import tensorflow as tf
import tensorflow.keras as keras
import input_data
import os
import numpy as np
from tqdm import trange

dataset = input_data.read_data_sets('mnist_data')
train_image = dataset.train

if not os.path.exists('gen_pic'):
    os.mkdir('gen_pic')

def gen_model():
    gen_input = keras.Input(shape=(100,))
    fc1 = keras.layers.Dense(256, activation='relu')(gen_input)
    gen_out = keras.layers.Dense(784, activation='sigmoid')(fc1)

    gen_model = keras.Model(gen_input, gen_out)
    return gen_model


def dis_model():
    dis_input = keras.Input(shape=(784,))
    fc1 = keras.layers.Dense(256, activation='relu')(dis_input)
    dis_out = keras.layers.Dense(1, activation='sigmoid')(fc1)

    dis_model = keras.Model(dis_input, dis_out)
    return dis_model


def gen_loss(dis_output):
    ones = tf.ones_like(dis_output)
    loss = keras.losses.binary_crossentropy(ones, dis_output)
    loss_sum = tf.reduce_mean(loss)
    return loss_sum


def dis_loss(dis_gen_out, dis_tar_out):
    # dis_gen_out 为判别器对生成器的预测输出
    # dis_tar_out 为判别器对真实数据的预测
    ones = tf.ones_like(dis_tar_out)
    zeros = tf.zeros_like(dis_gen_out)
    loss = tf.reduce_mean(keras.losses.binary_crossentropy(ones, dis_tar_out)) + \
           tf.reduce_mean(keras.losses.binary_crossentropy(zeros, dis_gen_out))
    return loss


optimizer = keras.optimizers.Adam(learning_rate=1e-4)

gen = gen_model()
dis = dis_model()

batch_size = 20

if os.path.exists('gen.h5'):
    gen.load_weights('gen.h5')
if os.path.exists('dis.h5'):
    dis.load_weights('dis.h5')

for step in trange(100000):
    input_data = np.random.normal(size=[batch_size, 100])  # 一部训练生成10个假样本
    true_image, _ = train_image.next_batch(batch_size // 2)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        gen_output = gen(input_data)
        dis_gen_out = dis(gen_output)
        dis_tar_out = dis(true_image)

        loss_gen = gen_loss(dis_gen_out)
        loss_dis = dis_loss(dis_gen_out, dis_tar_out)
    gen_grad = gen_tape.gradient(loss_gen, gen.trainable_variables)
    dis_grad = dis_tape.gradient(loss_dis, dis.trainable_variables)

    optimizer.apply_gradients(zip(gen_grad, gen.trainable_variables))
    optimizer.apply_gradients(zip(dis_grad, dis.trainable_variables))
    if step % 50 == 0:
        print('step: %d, gen loss: %.3f, dis loss: %.3f' % (step, loss_gen.numpy(), loss_dis.numpy()))
    if step % 500 == 0:
        img = gen_output[0].numpy().reshape((28, 28, 1))
        img = tf.keras.preprocessing.image.array_to_img(img * 255.0)
        img.save(os.path.join('gen_pic', 'generated_flog' + str(step) + '.png'))
        gen.save_weights('gen.h5')
        dis.save_weights('dis.h5')
