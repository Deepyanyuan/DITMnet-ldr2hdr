# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 10:17
# @Author  : Beny Liang
# @FileName: merge_HDR.py.py
# @Software: PyCharm Community Edition
# @gmail    ：liangb6228@gmail.com

'''
requests.cookies
~~~~~~~~~~~~~~~~
This code 
'''
import numpy as np
import cv2


def merge_HDR(y_dm_predict, y_um_predict, x_in, alpha):
    '''
    :param network_dm:
    :param network_um:
    :param x_in:
    :param alpha:
    :return:
    '''
    [batch_size, height_size, width_size, channels_size] = np.shape(x_in)
    # batch_size, height_size, width_size, channels_size = x_in.get_shape().as_list()
    y_dm_predict = np.reshape(y_dm_predict, (batch_size, -1, height_size, width_size, channels_size))
    y_um_predict = np.reshape(y_um_predict, (batch_size, -1, height_size, width_size, channels_size))

    y_dm_predict = np.flip(y_dm_predict, axis=1)
    x_base = np.add(y_dm_predict[:, -1, :, :, :], y_um_predict[:, 0, :, :, :]) / 2.0
    x_base = np.expand_dims(x_base, axis=1)
    # print('x_base.shape', np.shape(x_base))
    temp = np.concatenate([y_dm_predict, x_base], axis=1)
    out_img_list_total = np.concatenate([temp, y_um_predict], axis=1)
    # print('out_img_list_total.len', np.shape(out_img_list_total))

    merge_final_norm = list()
    merge_final = list()
    log_final = list()
    debevec_final = list()
    debevec_reverse = list()
    # new code
    for i in range(batch_size):
        # print('i', i)
        out_img_list = out_img_list_total[i].astype(np.float32)
        # log domain merge
        prev_img_log_mean = (out_img_list[7].astype(np.float32) + out_img_list[9].astype(
            np.float32)) / 2
        # b = np.log10(200 + 1)       # 对于Fairchild 200，NewHDR 27
        b = 1
        # up_log = np.power(10, out_img_list[9].astype(np.float32) * b) - 1
        pre_img_hdr = np.power(10, prev_img_log_mean * b) - 1  # 对数域恢复到原始域,仍然是归一化的

        # unit8 domain merge

        out_img_list = np.delete(out_img_list, [7, 9], axis=0)
        # print('out_img_list.len', len(out_img_list))

        out_img_list = (255. * out_img_list).astype(np.float32)
        threshold = 64  # 默认值为64
        stid = 0
        # print('out_img_list.len',len(out_img_list))
        prev_img = out_img_list[7].astype(np.float32)  # 从最中间图像选取作为预设标准
        out_img_list = np.flip(out_img_list, axis=0)  # 由明到暗
        for out_img in out_img_list[8:]:
            img = out_img.astype(np.float32)
            if (img > (prev_img + threshold)).sum() > 0:
                break
            prev_img = img[:, :, :]
            stid += 1

        edid = 0
        prev_img = out_img_list[7].astype(np.float32)
        out_img_list = np.flip(out_img_list, axis=0)  # 由暗到明
        for out_img in out_img_list[8:]:
            img = out_img.astype(np.float32)
            if (img < (prev_img - threshold)).sum() > 0:
                break
            prev_img = img[:, :, :]
            edid += 1

        inputs = list()
        out_img_list_ = out_img_list[7 - stid:8 + edid]  # 从中间向前后两侧选取阈值内的图像集
        exposure_times = list()
        lowest_exp_time = 1 / 32.  # 预设最小曝光时间，根据上面选取的图像数量进行曝光合成,默认值为1/1024
        for i in range(len(out_img_list_)):
            inputs.append(out_img_list_[i].astype(np.uint8))
            exposure_times.append(lowest_exp_time * np.power(np.sqrt(2.), i))
        exposure_times = np.array(exposure_times).astype(np.float32)
        # print('exposure_times.len',len(exposure_times))

        merge_debvec = cv2.createMergeDebevec()
        hdr_debvec = merge_debvec.process(inputs, times=exposure_times.copy())
        # hdr_debvec = hdr_debvec[..., ::-1]
        merge_final_debvec_temp = (1 - alpha) * pre_img_hdr / np.max(pre_img_hdr) + alpha * hdr_debvec / np.max(hdr_debvec)
        merge_final_norm.append(merge_final_debvec_temp)

        # merge_final_debvec_gamma = np.power(merge_final_debvec_temp, 2.2)
        # merge_final.append(merge_final_debvec_gamma)

        # log_final.append(pre_img_hdr)
        # log_final.append(up_log)
        # debevec_final.append(hdr_debvec_rgb)

    # return merge_final_norm, merge_final, log_final, debevec_final
    return merge_final_norm