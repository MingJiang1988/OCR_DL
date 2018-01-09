from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
#sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from rcnn import rcnn
from PIL import Image, ImageFilter


recognition = rcnn()


def mnist_tensor(image_path):
    os.system("python digits.py " + image_path)
    f = open("tmp/buff.txt", "r")
    sstr = f.read()
    f.close()
    return sstr  # first value in list


'''------------------------------------------------------------'''


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def getkey(item):
    return item[0]


def getboxkey(item):
    return item[1]


def sort_position_line(contour_list):
    #sorted(contour_list, key=getkey, reverse=True)
    contour_list.sort(key=getkey)
    return contour_list


def recognition_letters(img_path, switch):
    recognition_str = ""
    if switch == "d":
        recognition_str = recognition.get_string(img_path)
    if switch == "n":
        recognition_str = mnist_tensor(img_path)
    return recognition_str


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    i = 0
    imline = img.copy()
    boxlist = []
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        cv2.line(imline, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(imline, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(imline, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(imline, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        #min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        #min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        #max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        #max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        min_x = min(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        min_y = min(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
        max_x = max(int(box[0]), int(box[2]), int(box[4]), int(box[6]))
        max_y = max(int(box[1]), int(box[3]), int(box[5]), int(box[7]))
        het = max_y - min_y
        wei = max_x - min_x
        crop_img = img[min_y:max_y, min_x:max_x]
        boxlist.append([min_x, min_y, crop_img, het, wei])

    boxlist.sort(key=getboxkey)
    total_str = ""
    pos_y = 0
    buff_letters = []
    for i, b in enumerate(boxlist):
        nik_digits_scale = 0
        gray_img = cv2.cvtColor(b[2], cv2.COLOR_RGB2GRAY)
        blurred_size = 5
        blurred = cv2.GaussianBlur(gray_img, (blurred_size, blurred_size), 0)
        #wide = cv2.Canny(blurred, 10, 200)
        #tight = cv2.Canny(blurred, 225, 250)
        kernel_size = int(gray_img.shape[0]*0.2)
        if i==2 or i==3:
            kernel_size = int(gray_img.shape[0]*0.1)
            crop_scale = int(b[4] / b[3])
            if crop_scale > 10:
                nik_digits_scale = crop_scale

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_mask = auto_canny(blurred)
        img_dilation = cv2.dilate(edge_mask, kernel, iterations=1)
        contours_mask, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours_mask:
            x, y, w, h = cv2.boundingRect(ct)
            cv2.rectangle(img_dilation, (x, y), (x+w, y+h), (255, 255, 255), -1)

        cv2.imwrite("./tmp/Edges"+str(i)+".jpg", img_dilation)
        contours, hierarchy = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        j = 0
        contours_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > crop_img.shape[0]*0.5:
                split_mask = gray_img[y:y+h, x:x+w]
                file_path = "tmp/split"+str(i)+"_"+str(j)+".jpg"
                cv2.imwrite(file_path, split_mask)
                if nik_digits_scale == 0:
                    contours_list.append([x, y, w, h, file_path, "d"])
                if nik_digits_scale > 10:
                    contours_list.append([x, y, w, h, file_path, "n"])
                j += 1
        contours_list = sort_position_line(contours_list)
        letters = ""
        for split in contours_list:
            letters += recognition_letters(split[4], split[5]) + " "
        if (pos_y + b[3]*0.5) > b[1]:
            buff_letters.append([b[0], letters])
        if (pos_y + b[3]*0.5) < b[1]:
            a_line_str = ""
            if len(buff_letters) > 0:
                buff_letters.sort(key=getkey)
                for let in buff_letters:
                    a_line_str += " " + let[1]
                total_str = total_str + "\n" + a_line_str
                buff_letters = []
            if len(buff_letters) == 0:
                buff_letters.append([b[0], letters])
        pos_y = b[1]
        i += 1
        #line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
        #f.write(line)
    a_line_str = ""
    if len(buff_letters)>0:
        for lett in buff_letters:
            a_line_str += lett[1]
        total_str += "\n" + a_line_str
    cv2.imwrite("tmp/" + str(i) + ".jpg", crop_img)
    with open("output/"+base_name+".txt", "wb") as wf:
        wf.write(total_str)
    #img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("output", base_name), imline)


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


if __name__ == '__main__':
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.makedirs("tmp/")

    cfg_from_file('text.yml')
    config = tf.ConfigProto(allow_soft_placement=True)
    model_alpha = tf.Graph()
    model_digit = tf.Graph()

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)

