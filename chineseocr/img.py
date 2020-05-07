import os
from PIL import Image, ImageDraw
import numpy as np
from math import ceil, floor
from operator import itemgetter


def getFilesInDirect(path, str_dot_ext):
    """
    获取背景图像路径列表
    :param path: 背景图像存储路径. [str]
    :param str_dot_ext: 背景图像的存储格式. [str]
    :return:
    """
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.splitext(file_path)[1] == str_dot_ext:
            file_list.append(file_path)
    return file_list


def get_files_with_ext(path, str_ext):
    """
    获取具有str_ext结尾的文件路径列表
    :param path: 文件目录. [str]
    :param str_ext: 文件格式，如.png. [str]
    :return:
    """
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file_path.endswith(str_ext):
            file_list.append(file_path)
    return file_list


def get_target_txt_file(img_file):
    """
    加载对应图像的文本线文档对应的路径
    :param img_file: 图像路径. [str]
    :return:
    """
    # 或者文本线存储目录
    pre_dir = os.path.abspath(os.path.dirname(img_file) + os.path.sep + "..")
    txt_dir = os.path.join(pre_dir, 'contents')

    # 加载对应图像的文本线文档
    filename = os.path.basename(img_file)
    arr_split = os.path.splitext(filename)
    filename = arr_split[0] + '.txt'
    txt_file = os.path.join(txt_dir, filename)
    return txt_file


def get_list_contents(content_file):
    """
    获取文本线文档中的坐标和标签，并转化为列表，类似[[[1,2,3,4],'hello']]
    :param content_file: 文本线文档路径. [str]
    :return:
    """
    contents = []
    if not os.path.exists(content_file):
        return contents

    with open(content_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    for line in lines:
        arr_str = line.split('|')
        item = list(map(lambda x: int(x), arr_str[0].split(',')))
        contents.append([item, arr_str[1]])
    return contents


def get_image_and_targets(img_file, txt_file, anchor_heights):
    """
    加载图像，并获取图像卷积后的尺寸和真实标签
    :param img_file: 图像的路径. [str]
    :param txt_file: 图像对应的文本线文档路径. [str]
    :param anchor_heights: anchor高度列表. [list]
    :return:
    """
    # 加载图像
    img = Image.open(img_file)
    img_data = np.array(img, dtype=np.float32) / 255  # [height, width, channel]

    # 获取图像的rgb通道
    try:
        img_data = img_data[:, :, 0:3]
    except:
        img_data = img_data[:, :, 0:2]

    # 获取各个文本线的坐标和标注，并转化为列表
    txt_list = get_list_contents(txt_file)

    # targets
    img_size = img_data.shape  # height, width, channel

    # 计算卷积后feature map的高和宽
    height_feat = floor(ceil(ceil(img_size[0] / 2.0) / 2.0) / 3.0) - 2
    width_feat = ceil(ceil(ceil(img_size[1] / 2.0) / 2.0) / 2.0)

    # 初始化三个分支的目标值
    num_anchors = len(anchor_heights)
    target_cls = np.zeros((height_feat, width_feat, 2 * num_anchors))
    target_ver = np.zeros((height_feat, width_feat, 2 * num_anchors))
    target_hor = np.zeros((height_feat, width_feat, 2 * num_anchors))

    # 计算feature map上每个点的对应的真实标签
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width
    hc_start = 18
    wc_start = 4

    for h in range(height_feat):
        hc = hc_start + ash * h  # anchor height center
        for w in range(width_feat):
            cls, ver, hor = calculate_targets_at([hc, wc_start + asw * w], txt_list, anchor_heights)
            target_cls[h, w] = cls
            target_ver[h, w] = ver
            target_hor[h, w] = hor

    return [img_data], [height_feat, width_feat], target_cls, target_ver, target_hor


def calculate_targets_at(anchor_center, txt_list, anchor_heights):
    """
    计算当前anchor的真实标签
    :param anchor_center: anchor的中心,[height_center,width_center]. [list]
    :param txt_list: 文本线列表. [list]
    :param anchor_heights: anchor高度列表. [list]
    :return:
    """
    # anchor宽度和anchor高度、宽度步伐
    anchor_width = 8
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width

    # anchor中心
    hc = anchor_center[0]
    wc = anchor_center[1]

    # 初始化maxIoU和anchor_posi
    maxIoU = 0
    anchor_posi = 0
    text_bbox = []

    # 检测当前anchor是否包含文本，若存在，选择IoU最大的作为正例
    for item in txt_list:
        # 当前文本线的四个坐标
        bbox = item[0]

        flag = 0
        # 如果当前的anchor宽度中心刚好落在文本线内，则标记为1
        # 如果当前的文本线落在anchor宽度中心~anchor宽度中心+8范围内，并且比较靠近anchor宽度中心，则标记为1
        # 如果当前的文本线落在anchor宽度中心-8~anchor宽度中心范围内，并且比较靠近anchor宽度中心，则标记为1
        if bbox[0] < wc and wc <= bbox[2]:
            flag = 1
        elif wc < bbox[0] and bbox[2] < wc + asw:
            if bbox[0] - wc < wc + asw - bbox[2]:
                flag = 1
        elif wc - asw < bbox[0] and bbox[2] < wc:
            if bbox[2] - wc <= wc - asw - bbox[0]:
                flag = 1

        if flag == 0: continue

        # 文本线高度中心
        bcenter = (bbox[1] + bbox[3]) / 2.0

        # anchor的中心不能距离真实中心太远
        d0 = abs(hc - bcenter)
        dm = abs(hc - ash - bcenter)
        dp = abs(hc + ash - bcenter)

        if d0 < ash and d0 <= dm and d0 < dp:
            pass
        else:
            continue

        # 当检测到文本时，计算各个anchor的IoU，选择其中最大的作为正例
        posi = 0

        for ah in anchor_heights:
            hah = ah // 2  # half_ah

            IoU = 1.0 * (min(hc + hah, bbox[3]) - max(hc - hah, bbox[1])) \
                  / (max(hc + hah, bbox[3]) - min(hc - hah, bbox[1]))

            if IoU > maxIoU:
                maxIoU = IoU
                anchor_posi = posi
                text_bbox = bbox

            posi += 1
        break

    # 当检测不到文本时，三个分支的标签都用0表示
    if maxIoU <= 0:  #
        num_anchors = len(anchor_heights)
        cls = [0, 0] * num_anchors
        ver = [0, 0] * num_anchors
        hor = [0, 0] * num_anchors
        return cls, ver, hor

    # 检测出包含文本时，则最大IoU对应的anchor作为正例，其他作为负例
    cls = []
    ver = []
    hor = []
    for idx, ah in enumerate(anchor_heights):
        if not idx == anchor_posi:
            cls.extend([0, 0])
            ver.extend([0, 0])
            hor.extend([0, 0])
            continue
        cls.extend([1, 1])

        half_ah = ah // 2
        half_aw = anchor_width // 2

        # 计算anchor的绝对坐标
        anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]

        # 计算相对坐标，对anchor坐标进行修正
        ratio_bbox = [0, 0, 0, 0]
        ratio = (text_bbox[0] - anchor_bbox[0]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[0] = ratio

        ratio = (text_bbox[2] - anchor_bbox[2]) / anchor_width
        if abs(ratio) < 1:
            ratio_bbox[2] = ratio

        ratio_bbox[1] = (text_bbox[1] - anchor_bbox[1]) / ah
        ratio_bbox[3] = (text_bbox[3] - anchor_bbox[3]) / ah

        ver.extend([ratio_bbox[1], ratio_bbox[3]])
        hor.extend([ratio_bbox[0], ratio_bbox[2]])

    return cls, ver, hor


def trans_results(r_cls, r_ver, r_hor, anchor_heights, threshold):
    """
    将相对坐标转化为原始图像的绝对坐标，获取预测到的文本线坐标和分数
    :param r_cls: cls标签
    :param r_ver: ver标签
    :param r_hor: hor标签
    :param anchor_heights: anchor高度列表. [list]
    :param threshold: 分类阈值. [float]
    :return:
    """
    anchor_width = 8
    ash = 12  # anchor stride - height
    asw = 8  # anchor stride - width
    hc_start = 18
    wc_start = 4
    aw = anchor_width
    list_bbox = []
    list_conf = []
    feat_shape = r_cls.shape

    for h in range(feat_shape[0]):
        for w in range(feat_shape[1]):
            if max(r_cls[h, w, :]) < threshold:
                continue

            # 获取概率最大的anchor
            anchor_posi = np.argmax(r_cls[h, w, :])  # in r_cls
            anchor_id = anchor_posi // 2  # in anchor_heights

            # 计算anchor的坐标
            ah = anchor_heights[anchor_id]  #
            anchor_posi = anchor_id * 2  # for retrieve in r_ver, r_hor

            hc = hc_start + ash * h  # anchor center
            wc = wc_start + asw * w  # anchor center

            half_ah = ah // 2
            half_aw = aw // 2

            anchor_bbox = [wc - half_aw, hc - half_ah, wc + half_aw, hc + half_ah]

            # 计算预测到的文本线的坐标
            text_bbox = [0, 0, 0, 0]
            text_bbox[0] = anchor_bbox[0] + aw * r_hor[h, w, anchor_posi]
            text_bbox[1] = anchor_bbox[1] + ah * r_ver[h, w, anchor_posi]
            text_bbox[2] = anchor_bbox[2] + aw * r_hor[h, w, anchor_posi + 1]
            text_bbox[3] = anchor_bbox[3] + ah * r_ver[h, w, anchor_posi + 1]

            list_bbox.append(text_bbox)
            list_conf.append(max(r_cls[h, w, :]))

    return list_bbox, list_conf


def draw_text_boxes(img_file, text_bbox):
    """
    对图像绘制文本线
    :param img_file: 图像对应的路径. [str]
    :param text_bbox: 文本线坐标. [list]
    :return:
    """
    img_draw = Image.open(img_file)
    draw = ImageDraw.Draw(img_draw)
    for item in text_bbox:
        xs = item[0]
        ys = item[1]
        xe = item[2]
        ye = item[3]
        line_width = 1  # round(text_size/10.0)
        draw.line([(xs, ys), (xs, ye), (xe, ye), (xe, ys), (xs, ys)],
                  width=line_width, fill=(255, 0, 0))

    img_draw.save(img_file)


def do_nms_and_connection(list_bbox, list_conf):
    """将anchor连接为文本框
    :param list_bbox: anchor list,每个anchor包含左上右下四个坐标.[list]
    :param list_conf: anchor概率list，存放每个anchor为前景的概率，同list_bbox对应.[list]
    :return: 返回连接anchor后的文本框conn_bboxlist，每个文本框包含左上右下的四个坐标,[list]
    """
    # #设置anchor连接的最大距离，两个anchor距离大于50，则处理为两个文本框，反之则连接两个文本框
    # max_margin = 50
    # len_list_box = len(list_bbox)
    # conn_bbox = []
    # head = tail = 0
    # for i in range(1, len_list_box):
    #     distance_i_j = abs(list_bbox[i][0] - list_bbox[i - 1][0])
    #     overlap_i_j = overlap(list_bbox[i][1], list_bbox[i][3], list_bbox[i - 1][1], list_bbox[i - 1][3])
    #     if distance_i_j < max_margin and overlap_i_j > 0.7:
    #         tail = i
    #         if i == len_list_box - 1:
    #             this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
    #             conn_bbox.append(this_test_box)
    #             head = tail = i
    #     else:
    #         this_test_box = [list_bbox[head][0], list_bbox[head][1], list_bbox[tail][2], list_bbox[tail][3]]
    #         conn_bbox.append(this_test_box)
    #         head = tail = i

    # 获取每个anchor的近邻，判断条件是两个anchor之间的距离必须小于50个像素点，并且在垂直方向的重合度大于0.4
    neighbor_list = []
    for i in range(len(list_bbox) - 1):
        this_neighbor_list = [i]
        for j in range(i + 1, len(list_bbox)):
            distance_i_j = abs(list_bbox[i][2] - list_bbox[j][0])
            overlap_i_j = overlap(list_bbox[i][1], list_bbox[i][3], list_bbox[j][1], list_bbox[j][3])
            if distance_i_j < 50 and overlap_i_j > 0.4:
                this_neighbor_list.append(j)
        neighbor_list.append(this_neighbor_list)

    # 对每个近邻列表进行合并，一旦两个列表之间有共同的元素，则将他们并在一起
    conn_bbox = []
    while len(neighbor_list) > 0:
        this_conn_bbox = set(neighbor_list[0])
        filter_list = [0]
        for i in range(1, len(neighbor_list)):
            if len(this_conn_bbox & set(neighbor_list[i])) > 0:
                this_conn_bbox = this_conn_bbox | set(neighbor_list[i])
                filter_list.append(i)
        min_x = min([list_bbox[i][0] for i in list(this_conn_bbox)])
        min_y = np.mean([list_bbox[i][1] for i in list(this_conn_bbox)])
        max_x = max([list_bbox[i][2] for i in list(this_conn_bbox)])
        max_y = np.mean([list_bbox[i][3] for i in list(this_conn_bbox)])

        conn_bbox.append([min_x, min_y, max_x, max_y])
        neighbor_list = [neighbor_list[i] for i in range(len(neighbor_list)) if i not in filter_list]

    return conn_bbox


def overlap(h_up1, h_dw1, h_up2, h_dw2):
    """
    计算垂直重合度
    :param h_up1:
    :param h_dw1:
    :param h_up2:
    :param h_dw2:
    :return:
    """
    overlap_value = (min(h_dw1, h_dw2) - max(h_up1, h_up2)) \
                    / (max(h_dw1, h_dw2) - min(h_up1, h_up2))
    return overlap_value


def mean_gray(img):
    """图像灰度处理，均值法（多个通道的均值）
    :param img: img为通过cv2.imread()读入的图片
    :return: 均值法灰度化的图片数组
    """
    row, col, channel = img.shape
    img_gray = np.zeros(shape=(row, col))
    for r in range(row):
        for l in range(col):
            img_gray[r, l] = img[r, l, :].mean()

    return img_gray


def two_value_binary(img_gray, threshold=100, reverse=False):
    """
    二值法数据增强.
    :param img_gray: 灰度化后的图片数组.
    :param threshold: 二值化阈值, 大于阈值设为255， 小于阈值设为0.
    :param reverse:是否将前景和背景反转，默认False.[boolean]
    :return:
    """
    threshold /= 255
    img_binary = np.zeros_like(img_gray)
    row, col = img_binary.shape
    for i in range(row):
        for j in range(col):
            if img_gray[i, j] >= threshold:
                img_binary[i, j] = 1
            if reverse:
                img_binary[i, j] = 1 - img_binary[i, j]
    return img_binary


def convert2rgb(img_binary):
    """将二值化后图片改为三通道
    :param img_binary: 二值化后的图片，维度：二维.[numpy.ndarray]
    :return:
    """
    rows, cols = img_binary.shape
    img_binary_rgb = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            img_binary_rgb[i, j, 0:3] = np.tile(img_binary[i, j], 3)
    return img_binary_rgb
