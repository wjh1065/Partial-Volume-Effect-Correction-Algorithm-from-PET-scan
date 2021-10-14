import os
import matplotlib.pyplot as plt
from skimage import exposure
import nibabel as nib
from numpy import percentile

def get_data_list_input(root):
    dir_list = os.listdir(root)
    pet_list = []
    for i in dir_list:
        if i.startswith('ADNI'):
            print('------job Start------')
            data_path = os.path.join(root, i)
            subject_pvc_rPET = (data_path + '/pvc_rPET.nii.gz')
            subject_rPET = (data_path + '/gm_rPET.nii.gz')
            print('subject : ', subject_pvc_rPET)
            subject_img_header_pvc_rPET = nib.load(subject_pvc_rPET)
            subject_img_header_rPET = nib.load(subject_rPET)
            subject_img_pvc_rPET = subject_img_header_pvc_rPET.get_fdata()
            data_pvc_rPET = subject_img_header_pvc_rPET.get_fdata()

            subject_img_rPET = subject_img_header_rPET.get_fdata()
            data_rPET = subject_img_header_rPET.get_fdata()

            # plot basic intensity graph
            #hist1, bins_center1 = exposure.histogram(data)
            #plt.plot(bins_center1, hist1, lw=2)

            #plt.tight_layout()
            #plt.show()

            # step 1. calculate thr_triangle
            data_pvc_rPET = data_pvc_rPET[data_pvc_rPET > 0]


            q99 = percentile(data_pvc_rPET, 99)
            upper = q99
            print('upper :  ', upper)

            data_graph = data_pvc_rPET[data_pvc_rPET < upper]
            # plot basic intensity graph
            hist1, bins_center1 = exposure.histogram(data_graph)
            plt.plot(bins_center1, hist1, lw=2)

            plt.tight_layout()
            plt.show()

            # step 2. make thr lower upper threshold
            threshold_indices = subject_img_pvc_rPET <= 0
            subject_img_pvc_rPET[threshold_indices] = 0
            threshold_indices = subject_img_pvc_rPET >= upper
            subject_img_pvc_rPET[threshold_indices] = upper


            final_nii = nib.Nifti1Image(subject_img_pvc_rPET, subject_img_header_pvc_rPET.affine)
            final_nii.to_filename(os.path.join(data_path, 'method2_q99_pvc_rPET.nii.gz'))

            threshold_indices = subject_img_rPET <= 0
            subject_img_rPET[threshold_indices] = 0
            threshold_indices = subject_img_rPET >= upper
            subject_img_rPET[threshold_indices] = upper

            final_nii = nib.Nifti1Image(subject_img_rPET, subject_img_header_rPET.affine)
            final_nii.to_filename(os.path.join(data_path, 'method2_q99_gm_rPET.nii.gz'))

            print('------job End------')
            print('-------------------')
    return pet_list

root = '/home/wjh1065/Downloads/data/fsl_data/'
pet_list = get_data_list_input(root)