import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, ReLU, Cropping3D, LeakyReLU, PReLU, UpSampling3D, Activation,add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import nibabel as nib
import numpy as np
import shutil
import matplotlib.pyplot as plt
import gc


# 훈련 시작 및 종료 텔레그램 봇 실행
lcs_token = '1598459890:AAEEMip5BnlOU_XTqFRD8T6gD5q2ZUtHsLE'
lcs = telegram.Bot(token = lcs_token)

# 모델, 그래프, csv 파일 이름 설정하기
m = 'relu_1e-5_epoch500.h5'  # model name
l = 'relu_1e-5_epoch500.png' # Loss_graph name
c = 'relu_1e-5_epoch500.csv' # Loss_csv name

# 멀티 GPU 설정하기
strategy = tf.distribute.MirroredStrategy(
    ['/gpu:0'])  # select gpu num
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 패치 생성하기
def get_patches(img_arr, size=128, stride=128):
    patched_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping
        for i in range(i_max):
            for j in range(i_max):
                for k in range(i_max):
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size,
                                        k * stride: k * stride + size, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)



# 입력 데이터 폴더 리스트 불러오기 (Train)
def data_list_input(root):
    dir_list = os.listdir(root)
    list_input = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_input.append(data_path + '/Input_data.nii.gz')
    return list_input

# 정답 데이터 폴더 리스트 불러오기 (Train)
def data_list_output(root):
    dir_list = os.listdir(root)
    list_output = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_output.append(data_path + '/Target_data.nii.gz')
    return list_output

# 입력 데이터 폴더 리스트 불러오기 (Validation)
def data_list_val_input(root):
    dir_list = os.listdir(root)
    list_input = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_input.append(data_path + '/val_Input_data.nii.gz')
    return list_input

# 정답 데이터 폴더 리스트 불러오기 (Validation)
def data_list_val_output(root):
    dir_list = os.listdir(root)
    list_output = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_output.append(data_path + '/val_Target_data.nii.gz')
    return list_output

# 폴더 안에 있는 입력 데이터 패치 조각으로 불러오기 / normalize 실행 (0~1의 범위)
def data_load_input(file_list):
    patches = []
    for i in file_list:
        load_input = nib.load(i)
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        min_val = load_input.min()
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128, stride=128)
        patches.append(normalized_load_input)
    patches = np.vstack(patches)
    # print('input patches shape : ', patches.shape)
    return patches

# 폴더 안에 있는 정답 데이터 패치 조각으로 불러오기 / normalize 실행 (0~1의 범위)
def data_load_output(file_list):
    patches = []
    for i in file_list:
        load_output = nib.load(i)
        load_output = load_output.get_fdata()
        load_output = np.array(load_output, dtype=np.float32)
        max_val = load_output.max()
        min_val = load_output.min()
        normalized_load_output_1 = load_output / max_val
        normalized_load_output = get_patches(img_arr=normalized_load_output_1, size=128, stride=128)
        patches.append(normalized_load_output)
    patches = np.vstack(patches)
    # print('output patches shape : ', patches.shape)
    return patches

def data_load_val_input(file_list):
    patches = []
    for i in file_list:
        load_input = nib.load(i)
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        min_val = load_input.min()
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128, stride=128)
        patches.append(normalized_load_input)
    patches = np.vstack(patches)
    # print('val_input patches shape : ', patches.shape)
    return patches

def data_load_val_output(file_list):
    patches = []
    for i in file_list:
        load_output = nib.load(i)
        load_output = load_output.get_fdata()
        load_output = np.array(load_output, dtype=np.float32)
        max_val = load_output.max()
        min_val = load_output.min()
        normalized_load_output_1 = load_output / max_val
        normalized_load_output = get_patches(img_arr=normalized_load_output_1, size=128, stride=128)
        patches.append(normalized_load_output)
    patches = np.vstack(patches)
    # print('val_output patches shape : ', patches.shape)
    return patches


def Resunet3D_4_floor(filters=6):
    inputs = Input((128, 128, 128, 1))
    conv = Conv3D(filters * 2, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(conv)
    shortcut = Conv3D(filters * 4, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)
    output1 = add([shortcut, conv])

    res1 = BatchNormalization()(output1)
    res1 = Activation("relu")(res1)
    res1 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res1)
    res1 = BatchNormalization()(res1)
    res1 = Activation("relu")(res1)
    res1 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res1)
    shortcut1 = Conv3D(filters * 8, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output1)
    shortcut1 = BatchNormalization()(shortcut1)
    output2 = add([shortcut1, res1])

    res2 = BatchNormalization()(output2)
    res2 = Activation("relu")(res2)
    res2 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res2)
    res2 = BatchNormalization()(res2)
    res2 = Activation("relu")(res2)
    res2 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res2)
    shortcut2 = Conv3D(filters * 16, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output2)
    shortcut2 = BatchNormalization()(shortcut2)
    output3 = add([shortcut2, res2])

    res3 = BatchNormalization()(output3)
    res3 = Activation("relu")(res3)
    res3 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res3)
    res3 = BatchNormalization()(res3)
    res3 = Activation("relu")(res3)
    res3 = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res3)
    shortcut3 = Conv3D(filters * 32, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output3)
    shortcut3 = BatchNormalization()(shortcut3)
    output4 = add([shortcut3, res3])

    # bridge
    conv = BatchNormalization()(output4)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(conv)
    shortcut5 = Conv3D(filters * 64, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output4)
    shortcut5 = BatchNormalization()(shortcut5)
    output_bd = add([shortcut5, conv])

    # decoder
    uconv2 = UpSampling3D((2, 2, 2))(output_bd)
    uconv2 = concatenate([uconv2, output4])

    uconv22 = BatchNormalization()(uconv2)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv22)
    uconv22 = BatchNormalization()(uconv22)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv22)
    shortcut6 = Conv3D(filters * 16, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv2)
    shortcut6 = BatchNormalization()(shortcut6)
    output7 = add([uconv22, shortcut6])

    uconv3 = UpSampling3D((2, 2, 2))(output7)
    uconv3 = concatenate([uconv3, output3])

    uconv33 = BatchNormalization()(uconv3)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv33)
    uconv33 = BatchNormalization()(uconv33)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv33)
    shortcut7 = Conv3D(filters * 8, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv3)
    shortcut7 = BatchNormalization()(shortcut7)
    output8 = add([uconv33, shortcut7])

    uconv4 = UpSampling3D((2, 2, 2))(output8)
    uconv4 = concatenate([uconv4, output2])

    uconv44 = BatchNormalization()(uconv4)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv44)
    uconv44 = BatchNormalization()(uconv44)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv44)
    shortcut8 = Conv3D(filters * 4, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv4)
    shortcut8 = BatchNormalization()(shortcut8)
    output9 = add([uconv44, shortcut8])

    uconv5 = UpSampling3D((2, 2, 2))(output9)
    uconv5 = concatenate([uconv5, output1])

    uconv55 = BatchNormalization()(uconv5)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv55)
    uconv55 = BatchNormalization()(uconv55)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv3D(filters * 2, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv55)
    shortcut9 = Conv3D(filters * 2, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv5)
    shortcut9 = BatchNormalization()(shortcut9)
    output10 = add([uconv55, shortcut9])

    output_layer = Conv3D(1, (1, 1, 1), padding="same", activation="relu")(output10)
    model = Model(inputs, output_layer)

    return model

# 멀티GPU, 옵티마이져(Adam), 손실함수(MSE) 설정하기
with strategy.scope():
    model = Resunet3D_4_floor()
# model.summary()
opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss='mse')

# RAM 한계로 인해 1 epoch 진행 시 불러왔던 입력/정답데이터 삭제 후 다음 epoch에 다시 불러오기 반복.
for j in tqdm(range(1, 501)):  # epoch
    print('\n')
    print('epoch : ', j)
    print('\n-')

    # 첫번째 100개 입력/정답 데이터 불러오기
    root = './data/694_train_val_pred/100_train_1'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_100_1 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_100_1 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    model.fit(input_data_100_1, output_data_100_1, epochs=1, batch_size=4, shuffle=True, verbose=1)
    gc.collect()
    tf.keras.backend.clear_session()
    del input_data_100_1
    del output_data_100_1
    ##################################################

    # 두번째 100개 입력/정답 데이터 불러오기
    root = './data/694_train_val_pred/100_train_2'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_100_2 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_100_2 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    model.fit(input_data_100_2, output_data_100_2, epochs=1, batch_size=4, shuffle=True, verbose=1)
    gc.collect()
    tf.keras.backend.clear_session()
    del input_data_100_2
    del output_data_100_2
    ##################################################

    # 세번째 100개 입력/정답 데이터 불러오기
    root = './data/694_train_val_pred/100_train_3'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_100_3 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_100_3 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    model.fit(input_data_100_3, output_data_100_3, epochs=1, batch_size=4, shuffle=True, verbose=1)
    gc.collect()
    tf.keras.backend.clear_session()
    del input_data_100_3
    del output_data_100_3
    ##################################################

    # 네번째 100개 입력/정답 데이터 불러오기
    root = './data/694_train_val_pred/100_train_4'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_100_4 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_100_4 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    model.fit(input_data_100_4, output_data_100_4, epochs=1, batch_size=4, shuffle=True, verbose=1)
    gc.collect()
    tf.keras.backend.clear_session()

    del input_data_100_4
    del output_data_100_4
    ##################################################


    # 다섯번째 100개 입력/정답 데이터 불러오기
    root = './data/694_train_val_pred/100_train_5'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_100_5 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_100_5 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    model.fit(input_data_100_5, output_data_100_5, epochs=1, batch_size=4, shuffle=True, verbose=1)
    gc.collect()
    tf.keras.backend.clear_session()

    del input_data_100_5
    del output_data_100_5

    ##################################################

    # 마지막 50개 입력/정답 데이터와 검증 데이터 불러오기
    root = './data/694_train_val_pred/50_train'

    list_input = data_list_input(root)
    input_data_1 = np.array(data_load_input(list_input))
    del list_input
    input_data_50 = np.expand_dims(input_data_1, axis=4)
    del input_data_1
    gc.collect()

    list_output = data_list_output(root)
    output_data_1 = np.array(data_load_output(list_output))
    del list_output
    output_data_50 = np.expand_dims(output_data_1, axis=4)
    del output_data_1
    gc.collect()

    val_root = './data/694_train_val_pred/50_valid'

    list_val_input = data_list_val_input(val_root)
    valid_input_data_1 = np.array(data_load_val_input(list_val_input))
    del list_val_input
    valid_input_data = np.expand_dims(valid_input_data_1, axis=4)
    del valid_input_data_1
    gc.collect()

    list_val_output = data_list_val_output(val_root)
    valid_output_data_1 = np.array(data_load_val_output(list_val_output))
    del list_val_output
    valid_output_data = np.expand_dims(valid_output_data_1, axis=4)
    del valid_output_data_1
    gc.collect()

    history = model.fit(input_data_50, output_data_50, epochs=1, batch_size=4, shuffle=True, verbose=1,
                        validation_data=(valid_input_data, valid_output_data))
    gc.collect()
    tf.keras.backend.clear_session()
    del input_data_50
    del output_data_50
    del valid_input_data
    del valid_output_data

    # 특정 epoch 이상부터 loss graph 그리기
    if j >= 50:
        y_vloss = history.history['val_loss']
        y_loss = history.history['loss']
        x_len = j
        plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
        plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
        fig1 = plt.gcf()
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # save loss fig
        fig1.savefig('./results/1e-5_relu_epoch500.png', dpi=300)
    # 매 100 epoch 마다 모델 저장하기
    if j >= 100:
        model.save('./results/1e-5_relu_epoch100.h5')
    if j >= 200:
        model.save('./results/1e-5_relu_epoch200.h5')

model.save('model.h5')
shutil.move('model.h5', m)
shutil.move(m, 'results')


