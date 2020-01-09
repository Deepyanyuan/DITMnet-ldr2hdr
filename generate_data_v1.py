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
import os, glob, time, argparse, cv2, math, datetime
# from src.ops import camera_func_0, camera_func_1, camera_func_2, camera_func_3, camera_func_4, exposure_times, hdr_filter_func,hdr_filter_func1,hdr_filter_func2,hdr_filter_func3,hdr_filter_func4,hdr_filter_func5
from src.ops import *
# ----- Custom functions -----


# ----- Main function -----
if __name__ == '__main__':
    # Beginning running
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', help='Directory path of hdr images.', default='./hdrdata')    # HDR data dir
    parser.add_argument('-o', help='Directory path of ldr images.', default='./dataset/samples')  # output data dir

    args = parser.parse_args()

    # camera
    # func_dict = {'mark0': camera_func_0, 'mark1': camera_func_1, 'mark2': camera_func_2, 'mark3': camera_func_3, 'mark4': camera_func_4}
    # mark_list = ['mark0', 'mark1', 'mark2', 'mark3', 'mark4']

    func_dict = {'mark1': camera_func_1, 'mark2': camera_func_2, 'mark3': camera_func_3, 'mark4': camera_func_4}
    mark_list = ['mark1', 'mark2', 'mark3', 'mark4']

    tao = math.sqrt(2)
    T = 8
    normal_value = 3
    # file dir
    dir_in_path_list = glob.glob(args.i + '/*')
    dir_in_path_list = dir_in_path_list[:1]
    # max_hdrs = [200, 27]    # normalized HDR data sets
    # max_hdrs = [27]  # normalized HDR data sets
    max_hdrs = [200]  # normalized HDR data sets
    dir_out_path = glob.glob(args.o)
    Times = exposure_times(tao, T)


    N = len(dir_in_path_list)
    for i in range(N):
        print('第%d个文件夹' % i)

        dir_in_path = dir_in_path_list[i]
        filename_root = os.path.basename(dir_in_path)
        files_hdr_path_list = glob.glob(dir_in_path + '/*.hdr')
        current_hdr_max = max_hdrs[i]
        for file_num, file in enumerate(files_hdr_path_list):
            if file_num % 1 == 0:
                start = time.clock()
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
                # hdr_filter1 = hdr_filter_func1(hdr_norm)  # creating filter HDR data
                # hdr_filter2 = hdr_filter_func2(hdr_norm)  # creating filter HDR data
                # hdr_filter3 = hdr_filter_func3(hdr_norm)  # creating filter HDR data
                # hdr_filter4 = hdr_filter_func4(hdr_norm)  # creating filter HDR data
                # hdr_filter5 = hdr_filter_func5(hdr_norm)  # creating filter HDR data
                hdr_norm_exposure = list()
                hdr_filter_exposure = list()
                # hdr_filter_exposure1 = list()
                # hdr_filter_exposure2 = list()
                # hdr_filter_exposure3 = list()
                # hdr_filter_exposure4 = list()
                # hdr_filter_exposure5 = list()

                hdr_filter6 = hdr_filter_func6(hdr_norm)  # creating filter HDR data
                hdr_filter7 = hdr_filter_func7(hdr_norm)  # creating filter HDR data
                hdr_filter8 = hdr_filter_func8(hdr_norm)  # creating filter HDR data
                hdr_filter_exposure6 = list()
                hdr_filter_exposure7 = list()
                hdr_filter_exposure8 = list()


                for i in range(T + 1):
                    Time = Times[i]
                    hdr_norm_exposure.append(hdr_norm * Time)
                    hdr_filter_exposure.append(hdr_filter * Time)
                    # hdr_filter_exposure1.append(hdr_filter1 * Time)
                    # hdr_filter_exposure2.append(hdr_filter2 * Time)
                    # hdr_filter_exposure3.append(hdr_filter3 * Time)
                    # hdr_filter_exposure4.append(hdr_filter4 * Time)
                    # hdr_filter_exposure5.append(hdr_filter5 * Time)

                    hdr_filter_exposure6.append(hdr_filter6 * Time)
                    hdr_filter_exposure7.append(hdr_filter7 * Time)
                    hdr_filter_exposure8.append(hdr_filter8 * Time)

                hdr_norm_exposure = np.array(hdr_norm_exposure)
                hdr_filter_exposure = np.array(hdr_filter_exposure)
                # hdr_filter_exposure1 = np.array(hdr_filter_exposure1)
                # hdr_filter_exposure2 = np.array(hdr_filter_exposure2)
                # hdr_filter_exposure3 = np.array(hdr_filter_exposure3)
                # hdr_filter_exposure4 = np.array(hdr_filter_exposure4)
                # hdr_filter_exposure5 = np.array(hdr_filter_exposure5)

                hdr_filter_exposure6 = np.array(hdr_filter_exposure6)
                hdr_filter_exposure7 = np.array(hdr_filter_exposure7)
                hdr_filter_exposure8 = np.array(hdr_filter_exposure8)

                for i in range(len(mark_list)):
                    mark = mark_list[i]
                    ldr_norm_temp = func_dict[mark](hdr_norm_exposure)
                    end = datetime.datetime.now()
                    ldr_filter_temp = func_dict[mark](hdr_filter_exposure)
                    # ldr_filter_temp1 = func_dict[mark](hdr_filter_exposure1)
                    # ldr_filter_temp2 = func_dict[mark](hdr_filter_exposure2)
                    # ldr_filter_temp3 = func_dict[mark](hdr_filter_exposure3)
                    # ldr_filter_temp4 = func_dict[mark](hdr_filter_exposure4)
                    # ldr_filter_temp5 = func_dict[mark](hdr_filter_exposure5)

                    ldr_filter_temp6 = func_dict[mark](hdr_filter_exposure6)
                    ldr_filter_temp7 = func_dict[mark](hdr_filter_exposure7)
                    ldr_filter_temp8 = func_dict[mark](hdr_filter_exposure8)
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
                    # cut_ldr_temp_11 = ldr_filter_temp1[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    # cut_ldr_temp_12 = ldr_filter_temp2[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    # cut_ldr_temp_13 = ldr_filter_temp3[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    # cut_ldr_temp_14 = ldr_filter_temp4[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    # cut_ldr_temp_15 = ldr_filter_temp5[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image

                    cut_ldr_temp_16 = ldr_filter_temp6[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    cut_ldr_temp_17 = ldr_filter_temp7[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image
                    cut_ldr_temp_18 = ldr_filter_temp8[:, 0:img_patch, 0:img_patch, :]  # creating filtered LDR image

                    cut_ldr_temp_2 = ldr_norm_temp[:, height-img_patch:height, width-img_patch:width, :]
                    cut_ldr_temp_3 = ldr_filter_temp[:, height-img_patch:height, width-img_patch:width, :]
                    # cut_ldr_temp_31 = ldr_filter_temp1[:, height - img_patch:height, width - img_patch:width, :]
                    # cut_ldr_temp_32 = ldr_filter_temp2[:, height - img_patch:height, width - img_patch:width, :]
                    # cut_ldr_temp_33 = ldr_filter_temp3[:, height - img_patch:height, width - img_patch:width, :]
                    # cut_ldr_temp_34 = ldr_filter_temp4[:, height - img_patch:height, width - img_patch:width, :]
                    # cut_ldr_temp_35 = ldr_filter_temp5[:, height - img_patch:height, width - img_patch:width, :]

                    cut_ldr_temp_36 = ldr_filter_temp6[:, height - img_patch:height, width - img_patch:width, :]
                    cut_ldr_temp_37 = ldr_filter_temp7[:, height - img_patch:height, width - img_patch:width, :]
                    cut_ldr_temp_38 = ldr_filter_temp8[:, height - img_patch:height, width - img_patch:width, :]


                    re_size = (256, 256)
                    # shrink_cut_hdr_temp_0 = cv2.resize(cut_hdr_temp_0, re_size, interpolation=cv2.INTER_AREA)
                    # shrink_cut_hdr_temp_1 = cv2.resize(cut_hdr_temp_1, re_size, interpolation=cv2.INTER_AREA)
                    # shrink_cut_hdr_temp_2 = cv2.resize(cut_hdr_temp_2, re_size, interpolation=cv2.INTER_AREA)
                    # shrink_cut_hdr_temp_3 = cv2.resize(cut_hdr_temp_3, re_size, interpolation=cv2.INTER_AREA)

                    # save output data dir
                    savepath_1 = save_root_path + '_sub1'
                    savepath_2 = save_root_path + '_sub2'
                    class_H_path_1 = savepath_1 + '/HDR'
                    class_L_path_1 = savepath_1 + '/LDR'
                    class_F_path_1 = savepath_1 + '/FLDR'
                    class_F_path_11 = savepath_1 + '/FLDR1'
                    class_F_path_12 = savepath_1 + '/FLDR2'
                    class_F_path_13 = savepath_1 + '/FLDR3'
                    class_F_path_14 = savepath_1 + '/FLDR4'
                    class_F_path_15 = savepath_1 + '/FLDR5'
                    class_F_path_16 = savepath_1 + '/FLDR6'
                    class_F_path_17 = savepath_1 + '/FLDR7'
                    class_F_path_18 = savepath_1 + '/FLDR8'
                    class_H_path_2 = savepath_2 + '/HDR'
                    class_L_path_2 = savepath_2 + '/LDR'
                    class_F_path_2 = savepath_2 + '/FLDR'
                    class_F_path_21 = savepath_2 + '/FLDR1'
                    class_F_path_22 = savepath_2 + '/FLDR2'
                    class_F_path_23 = savepath_2 + '/FLDR3'
                    class_F_path_24 = savepath_2 + '/FLDR4'
                    class_F_path_25 = savepath_2 + '/FLDR5'
                    class_F_path_26 = savepath_2 + '/FLDR6'
                    class_F_path_27 = savepath_2 + '/FLDR7'
                    class_F_path_28 = savepath_2 + '/FLDR8'
                    os.makedirs(class_H_path_1)
                    os.makedirs(class_L_path_1)
                    os.makedirs(class_F_path_1)
                    os.makedirs(class_F_path_11)
                    os.makedirs(class_F_path_12)
                    os.makedirs(class_F_path_13)
                    os.makedirs(class_F_path_14)
                    os.makedirs(class_F_path_15)
                    os.makedirs(class_F_path_16)
                    os.makedirs(class_F_path_17)
                    os.makedirs(class_F_path_18)
                    os.makedirs(class_H_path_2)
                    os.makedirs(class_L_path_2)
                    os.makedirs(class_F_path_2)
                    os.makedirs(class_F_path_21)
                    os.makedirs(class_F_path_22)
                    os.makedirs(class_F_path_23)
                    os.makedirs(class_F_path_24)
                    os.makedirs(class_F_path_25)
                    os.makedirs(class_F_path_26)
                    os.makedirs(class_F_path_27)
                    os.makedirs(class_F_path_28)
                    # cv2.imwrite(class_H_path_1 + '/0.hdr', shrink_cut_hdr_temp_0)
                    # cv2.imwrite(class_H_path_1 + '/1.hdr', shrink_cut_hdr_temp_1)
                    # cv2.imwrite(class_H_path_2 + '/0.hdr', shrink_cut_hdr_temp_2)
                    # cv2.imwrite(class_H_path_2 + '/1.hdr', shrink_cut_hdr_temp_3)

                    for n in range(exposure_N):
                        shrink_cut_ldr_temp_0 = cv2.resize(cut_ldr_temp_0[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_1 = cv2.resize(cut_ldr_temp_1[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_11 = cv2.resize(cut_ldr_temp_11[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_12 = cv2.resize(cut_ldr_temp_12[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_13 = cv2.resize(cut_ldr_temp_13[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_14 = cv2.resize(cut_ldr_temp_14[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_15 = cv2.resize(cut_ldr_temp_15[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_16 = cv2.resize(cut_ldr_temp_16[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_17 = cv2.resize(cut_ldr_temp_17[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_18 = cv2.resize(cut_ldr_temp_18[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_2 = cv2.resize(cut_ldr_temp_2[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_3 = cv2.resize(cut_ldr_temp_3[n] * 255, re_size,
                                                           interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_31 = cv2.resize(cut_ldr_temp_31[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_32 = cv2.resize(cut_ldr_temp_32[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_33 = cv2.resize(cut_ldr_temp_33[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_34 = cv2.resize(cut_ldr_temp_34[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        # shrink_cut_ldr_temp_35 = cv2.resize(cut_ldr_temp_35[n] * 255, re_size,
                        #                                    interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_36 = cv2.resize(cut_ldr_temp_36[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_37 = cv2.resize(cut_ldr_temp_37[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_38 = cv2.resize(cut_ldr_temp_38[n] * 255, re_size,
                                                            interpolation=cv2.INTER_AREA)
                        cv2.imwrite(class_L_path_1 + '/' + str(n) + '.png', shrink_cut_ldr_temp_0)
                        cv2.imwrite(class_F_path_1 + '/' + str(n) + '.png', shrink_cut_ldr_temp_1)
                        # cv2.imwrite(class_F_path_11 + '/' + str(n) + '.png', shrink_cut_ldr_temp_11)
                        # cv2.imwrite(class_F_path_12 + '/' + str(n) + '.png', shrink_cut_ldr_temp_12)
                        # cv2.imwrite(class_F_path_13 + '/' + str(n) + '.png', shrink_cut_ldr_temp_13)
                        # cv2.imwrite(class_F_path_14 + '/' + str(n) + '.png', shrink_cut_ldr_temp_14)
                        # cv2.imwrite(class_F_path_15 + '/' + str(n) + '.png', shrink_cut_ldr_temp_15)
                        cv2.imwrite(class_F_path_16 + '/' + str(n) + '.png', shrink_cut_ldr_temp_16)
                        cv2.imwrite(class_F_path_17 + '/' + str(n) + '.png', shrink_cut_ldr_temp_17)
                        cv2.imwrite(class_F_path_18 + '/' + str(n) + '.png', shrink_cut_ldr_temp_18)
                        cv2.imwrite(class_L_path_2 + '/' + str(n) + '.png', shrink_cut_ldr_temp_2)
                        cv2.imwrite(class_F_path_2 + '/' + str(n) + '.png', shrink_cut_ldr_temp_3)
                        # cv2.imwrite(class_F_path_21 + '/' + str(n) + '.png', shrink_cut_ldr_temp_31)
                        # cv2.imwrite(class_F_path_22 + '/' + str(n) + '.png', shrink_cut_ldr_temp_32)
                        # cv2.imwrite(class_F_path_23 + '/' + str(n) + '.png', shrink_cut_ldr_temp_33)
                        # cv2.imwrite(class_F_path_24 + '/' + str(n) + '.png', shrink_cut_ldr_temp_34)
                        # cv2.imwrite(class_F_path_25 + '/' + str(n) + '.png', shrink_cut_ldr_temp_35)
                        cv2.imwrite(class_F_path_26 + '/' + str(n) + '.png', shrink_cut_ldr_temp_36)
                        cv2.imwrite(class_F_path_27 + '/' + str(n) + '.png', shrink_cut_ldr_temp_37)
                        cv2.imwrite(class_F_path_28 + '/' + str(n) + '.png', shrink_cut_ldr_temp_38)

                end = time.clock()
                print(end - start)
    # ----- Program over
    print('success! u are a smart boy!')
