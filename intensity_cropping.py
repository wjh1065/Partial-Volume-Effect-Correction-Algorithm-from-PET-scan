import os
import matplotlib.pyplot as plt
from skimage import exposure
import nibabel as nib
from numpy import percentile


def intensity_cropping(root):
    # root폴더에 있는 영상 폴더 목록 가져오기
    dir_list = os.listdir(root)
    pet_list = []
    for i in dir_list:
        if i.startswith('ADNI'):
            print('Intensity cropping job Start')
            data_path = os.path.join(root, i)

            # 입력 데이터 rPET.nii.gz 불러오기
            # 정답 데이터 PVC_rPET.nii.gz 불러오기
            subject_pvc_rPET = (data_path + '/pvc_rPET.nii.gz')
            subject_rPET = (data_path + '/gm_rPET.nii.gz')

            # 각 데이터들의 NIFTI 헤더 정보 추출
            subject_img_header_pvc_rPET = nib.load(subject_pvc_rPET)
            subject_img_header_rPET = nib.load(subject_rPET)

            # 뇌 영상 데이터(nii.gz)를 numpy 형식으로 가져오기
            subject_img_pvc_rPET = subject_img_header_pvc_rPET.get_fdata()
            data_pvc_rPET = subject_img_header_pvc_rPET.get_fdata()

            subject_img_rPET = subject_img_header_rPET.get_fdata()
            data_rPET = subject_img_header_rPET.get_fdata()

            # step 1. 음수 값 자르기,0% ~ 99% 범위 설정하기
            data_pvc_rPET = data_pvc_rPET[data_pvc_rPET > 0]
            q99 = percentile(data_pvc_rPET, 99)
            upper = q99

            # step 2. 0 이상 상위 99% 이상은 99%의 값으로 설정하기
            threshold_indices = subject_img_pvc_rPET <= 0
            subject_img_pvc_rPET[threshold_indices] = 0
            threshold_indices = subject_img_pvc_rPET >= upper
            subject_img_pvc_rPET[threshold_indices] = upper

            final_nii = nib.Nifti1Image(subject_img_pvc_rPET, subject_img_header_pvc_rPET.affine)
            final_nii.to_filename(os.path.join(data_path, 'Target_data.nii.gz'))

            threshold_indices = subject_img_rPET <= 0
            subject_img_rPET[threshold_indices] = 0
            threshold_indices = subject_img_rPET >= upper
            subject_img_rPET[threshold_indices] = upper

            final_nii = nib.Nifti1Image(subject_img_rPET, subject_img_header_rPET.affine)
            final_nii.to_filename(os.path.join(data_path, 'Input_data.nii.gz'))
            print('job End')
    return pet_list


root = '/home/wjh1065/Downloads/data/fsl_data/'
pet_list = get_data_list_input(root)
