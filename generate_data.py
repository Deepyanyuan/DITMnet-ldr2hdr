# -*- encoding: utf-8 -*-
from __future__ import division

# ----- author info
"""
@File    : generate_data.py
@Time    : 2019/11/16 20:41
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description: 

'''
# ----- Load libraries 
import numpy as np
import os, glob, time, argparse, cv2, math
from src.ops import camera_func_0, camera_func_1, camera_func_2, camera_func_3, camera_func_4, exposure_times, hdr_filter_func,hdr_filter_func1,hdr_filter_func2,hdr_filter_func3,hdr_filter_func4,hdr_filter_func5

# ----- Custom functions -----


# ----- Main function -----
if __name__ == '__main__':
    # Beginning running
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', help='Directory path of hdr images.', default='./hdrdata')    # HDR data dir
    parser.add_argument('-o', help='Directory path of ldr images.', default='./dataset/samples')  # output data dir

    args = parser.parse_args()

    # camera
    func_dict = {'mark0': camera_func_0, 'mark1': camera_func_1, 'mark2': camera_func_2, 'mark3': camera_func_3, 'mark4': camera_func_4}
    mark_list = ['mark0', 'mark1', 'mark2', 'mark3', 'mark4']

    tao = math.sqrt(2)
    T = 8
    normal_value = 3
    # file dir
    dir_in_path_list = glob.glob(args.i + '/*')
    dir_in_path_list = dir_in_path_list[1:]
    # max_hdrs = [200, 27]    # normalized HDR data sets
    max_hdrs = [27]  # normalized HDR data sets
    dir_out_path = glob.glob(args.o)
    Times = exposure_times(tao, T)

    start = time.clock()
    N = len(dir_in_path_list)
    for i in range(N):
        print('第%d个文件夹' % i)
        dir_in_path = dir_in_path_list[i]
        filename_root = os.path.basename(dir_in_path)
        files_hdr_path_list = glob.glob(dir_in_path + '/*.hdr')
        current_hdr_max = max_hdrs[i]
        for file_num, file in enumerate(files_hdr_path_list):
            if file_num % 1 == 0:
                hdr = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH)  # imread HDR data, note that it is BGR format
                hdr_0 = hdr + (10 ** -8)
                filename_hdr, file_format = os.path.splitext(file)
                filename_sub = os.path.basename(filename_hdr)
                print('file name:', filename_sub)
                hdr_log = np.log10(hdr_0 + 1)
                hdr_log_norm = (hdr_log - np.min(hdr_log)) / (np.log10(current_hdr_max + 1))  # normalized log-domain HDR data
                # hdr_mean = np.mean(hdr_0)
                # hdr_max = np.max(hdr_0)
                hdr_norm = (hdr_0 - np.min(hdr_0)) / (4 * np.mean(hdr_0) - np.min(hdr_0))
                hdr_0_norm = hdr_0 / current_hdr_max

                hdr_filter = hdr_filter_func(hdr_norm)  # creating filter HDR data
                hdr_norm_exposure = list()
                hdr_filter_exposure = list()
                for i in range(T + 1):
                    Time = Times[i]
                    hdr_norm_exposure.append(hdr_norm * Time)
                    hdr_filter_exposure.append(hdr_filter * Time)
                hdr_norm_exposure = np.array(hdr_norm_exposure)
                hdr_filter_exposure = np.array(hdr_filter_exposure)

                for i in range(len(mark_list)):
                    # print('camera function %d'%i)
                    # start = datetime.datetime.now()
                    # print('start time',start)
                    mark = mark_list[i]
                    ldr_norm_temp = func_dict[mark](hdr_norm_exposure)
                    # print(hdr_norm_temp-hdr_norm_exposure)
                    # end = datetime.datetime.now()
                    # print('end time',end)
                    # print('hdr_norm_temp is ',end - start)
                    # start_1 = datetime.datetime.now()
                    ldr_filter_temp = func_dict[mark](hdr_filter_exposure)
                    # end_1 = datetime.datetime.now()
                    # print('hdr_filter_temp is ', end_1 - start_1)
                    save_root_path = dir_out_path[0] + '/' + filename_root + '_' + filename_sub + '_' + mark

                    # 裁剪图片
                    count = 0
                    image_each = 2
                    exposure_N, height, width, channel = np.shape(ldr_norm_temp)
                    img_patch = np.min([height, width])

                    cut_hdr_temp_0 = hdr_log_norm[0:img_patch, 0:img_patch, :]
                    cut_hdr_temp_1 = hdr_0[0:img_patch, 0:img_patch, :]  # original HDR data
                    cut_hdr_temp_2 = hdr_log_norm[height - img_patch:height, width - img_patch:width, :]
                    cut_hdr_temp_3 = hdr_0[height - img_patch:height, width - img_patch:width, :]

                    cut_ldr_temp_0 = ldr_norm_temp[:, 0:img_patch, 0:img_patch, :]  # creating traditional LDR images
                    cut_ldr_temp_1 = ldr_filter_temp[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    cut_ldr_temp_2 = ldr_norm_temp[:, height-img_patch:height, width-img_patch:width, :]
                    cut_ldr_temp_3 = ldr_filter_temp[:, height-img_patch:height, width-img_patch:width, :]

                    re_size = (256, 256)
                    shrink_cut_hdr_temp_0 = cv2.resize(cut_hdr_temp_0, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_1 = cv2.resize(cut_hdr_temp_1, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_2 = cv2.resize(cut_hdr_temp_2, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_3 = cv2.resize(cut_hdr_temp_3, re_size, interpolation=cv2.INTER_AREA)

                    # save output data dir
                    savepath_1 = save_root_path + '_sub1'
                    savepath_2 = save_root_path + '_sub2'
                    class_H_path_1 = savepath_1 + '/HDR'
                    class_L_path_1 = savepath_1 + '/LDR'
                    class_F_path_1 = savepath_1 + '/FLDR'
                    class_H_path_2 = savepath_2 + '/HDR'
                    class_L_path_2 = savepath_2 + '/LDR'
                    class_F_path_2 = savepath_2 + '/FLDR'
                    os.makedirs(class_H_path_1)
                    os.makedirs(class_L_path_1)
                    os.makedirs(class_F_path_1)
                    os.makedirs(class_H_path_2)
                    os.makedirs(class_L_path_2)
                    os.makedirs(class_F_path_2)
                    cv2.imwrite(class_H_path_1 + '/0.hdr', shrink_cut_hdr_temp_0)
                    cv2.imwrite(class_H_path_1 + '/1.hdr', shrink_cut_hdr_temp_1)
                    cv2.imwrite(class_H_path_2 + '/0.hdr', shrink_cut_hdr_temp_2)
                    cv2.imwrite(class_H_path_2 + '/1.hdr', shrink_cut_hdr_temp_3)

                    for n in range(exposure_N):
                        shrink_cut_ldr_temp_0 = cv2.resize(cut_ldr_temp_0[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_1 = cv2.resize(cut_ldr_temp_1[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_2 = cv2.resize(cut_ldr_temp_2[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_3 = cv2.resize(cut_ldr_temp_3[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        cv2.imwrite(class_L_path_1 + '/' + str(n) + '.png', shrink_cut_ldr_temp_0)
                        cv2.imwrite(class_F_path_1 + '/' + str(n) + '.png', shrink_cut_ldr_temp_1)
                        cv2.imwrite(class_L_path_2 + '/' + str(n) + '.png', shrink_cut_ldr_temp_2)
                        cv2.imwrite(class_F_path_2 + '/' + str(n) + '.png', shrink_cut_ldr_temp_3)

    end = time.clock()
    print(end - start)
    # ----- Program over
    print('success! u are a smart boy!')
