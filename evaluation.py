import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tqdm import tqdm
import nibabel as nib
import numpy as np
import time
import os
import math
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from skimage.metrics import mean_squared_error as mse
# from skimage.metrics import mean


os.environ["CUDA_VISIBLE_DEVICES"]='0'

"""
make patch
"""
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
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size, k * stride: k * stride + size, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)

"""
reconstruct patch data
"""
def reconstruct_patch(img_arr, org_img_size, stride=128, size=128):
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")
    if size is None:
        size = img_arr.shape[2]
    if stride is None:
        stride = size
    nm_layers = img_arr.shape[4]
    i_max = (org_img_size[0] // stride ) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride ) + 1 - (size // stride)
    k_max = (org_img_size[2] // stride ) + 1 - (size // stride)
    total_nm_images = img_arr.shape[0] // (i_max ** 3)
    images_list = []
    kk=0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0],org_img_size[1],org_img_size[2],nm_layers), dtype=img_arr[0].dtype)
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for layer in range(nm_layers):
                        img_bg[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * stride: k * stride + size,
                        layer,
                        ] = img_arr[kk, :, :, :, layer]
                    kk += 1
        images_list.append(img_bg)
    return np.stack(images_list)


"""
load data //  pred LR to HR
"""


def pred_LR2HR(root):
    dir_list = os.listdir(root)
    pred_list_LR = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            pred_list_LR.append(data_path + '/method2_q99_gm_rPET.nii.gz')
    return pred_list_LR

"""
load patch pred LR data 
"""

def data_load_pred_LR(root, file_list):
    for i in tqdm(file_list):
        load_input = nib.load(i)
        print('subject : ',i[30:67])
        
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        #print('input_data max : ', max_val)
        min_val = load_input.min()
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128,stride=128)
        #print('patches shape : ', normalized_load_input.shape)
        pred_data = np.expand_dims(normalized_load_input,axis=4)
        #print('pred data shape : ', pred_data.shape)
        
        model = load_model('./results/300/1e-5_relu_epoch300.h5')
        
        pred = model.predict(pred_data, batch_size=1)
        #print('model predict done')
        x_reconstructed = reconstruct_patch(img_arr=pred, org_img_size=(256,256,256), stride=128)
        final_pred = np.squeeze(x_reconstructed)
        #print('reconstructed shape : ', x_reconstructed.shape)
        header = nib.load(i)
        header_1 = header.get_fdata()
        max_val = header_1.max()
        final_pred_1 = final_pred * max_val
        
        data_path = os.path.join(root, i[39:56])
        
        img = nib.Nifti1Image(final_pred_1, header.affine)
        img.to_filename(os.path.join(data_path, '128_Vnet_relu_epoch300.nii.gz'))
        print('pred save done')
    return img

def psnr(img1, img2, max_val):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse_function(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return math.sqrt(mse)


def psnr_ssim(root, file_list):
    results = []
    for i in tqdm(file_list):
        load_subject = nib.load(i)

        print('subject : ', i[34:71])
        data_path = os.path.join(root, i[34:71])
        
        target_data = os.path.join(data_path, 'method2_q99_pvc_rPET.nii.gz')
        input_data = os.path.join(data_path, 'method2_q99_gm_rPET.nii.gz')
        sharpen_data = os.path.join(data_path, 'sharpened_method2_q99_gm_rPET.nii.gz')
        pred_data = os.path.join(data_path, '128_Vnet_relu_epoch300.nii.gz')
        #print('target data : ', target_data)
        #print('input data : ', input_data)
        #print('pred data : ', pred_data)
        
        target_data = nib.load(target_data)
        target_data = target_data.get_fdata()
        target_max_val = target_data.max()
        
        input_data = nib.load(input_data)
        input_data = input_data.get_fdata()
        input_max_val = input_data.max()

        sharpen_data = nib.load(sharpen_data)
        sharpen_data = sharpen_data.get_fdata()
        sharpen_max_val = sharpen_data.max()
        
        pred_data = nib.load(pred_data)
        pred_data = pred_data.get_fdata()
        pred_max_val = pred_data.max()
        
        """
        psnr step
        """
        target_psnr = mse_function(target_data, target_data)
        target_ssim = ssim(target_data, target_data, data_range=target_max_val)
        
        input_mse = np.sqrt(mse(target_data, input_data))
        input_ssim = ssim(target_data, input_data, data_range=input_max_val)

        sharpen_mse = np.sqrt(mse(target_data, sharpen_data))
        sharpen_ssim = ssim(target_data, sharpen_data, data_range=sharpen_max_val)
        
        pred_mse = np.sqrt(mse(target_data, pred_data))
        pred_ssim = ssim(target_data, pred_data, data_range=pred_max_val)
        
        #print('-'*(50))
        # print('subject : ',i[37:54]) # 39 56
        
        #print('psnr (target , target) : ','%.4f' % target_psnr)
        #print('psnr (target , input) : ','%.4f' % input_psnr)
        #print('----- psnr (target , pred) ----- : ','%.4f' % pred_psnr)
        
        #print('ssim (target , target) : ','%.4f' % target_ssim)
        #print('ssim (target , target) : ','%.4f' % input_ssim)
        #print('----- ssim (target , pred) ----- : ','%.4f' % pred_ssim)
        result = [i[34:71], round(input_mse,4),round(sharpen_mse,4),round(pred_mse,4), round(input_ssim,4),round(sharpen_ssim,4),round(pred_ssim,4)]
        print('Result : \n' ,result)
        #print('-'*(50))
        results.append(result)
        # print(results)
        hist_df = pd.DataFrame(results)
        # save [mse, ssim] csv
        hist_csv_file = 'result_ADNI_PET_sharpen_rmse_ssim.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
    return result



root = './data/694_train_val_pred/94_pred'

pred_list_LR = pred_LR2HR(root)

# pred_LR = np.array(data_load_pred_LR(root,pred_list_LR))


result = psnr_ssim(root,pred_list_LR)


