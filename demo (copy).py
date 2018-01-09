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


'''-----------------------------------------------------------'''


def predictint(imvalue):
    import tensorflow as tf

    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        try:
            saver.restore(sess, "./nmodel/model2.ckpt")
        except Exception as e:
            print(e)
        prediction = tf.argmax(y_conv, 1)
        return prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheigth == 0):  # rare case but minimum is 1 pixel
            nheigth = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
    # print(tva)


def mnist_tensor(image_path):
    imvalue = imageprepare(image_path)
    predint = predictint(imvalue)
    return predint[0]  # first value in list


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
    #with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
    #dim = (640, 383)
    #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
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
        kernel_size = int(gray_img.shape[0]*0.3)
        if i==2 or i==3:
            kernel_size = int(gray_img.shape[0]*0.1)
            crop_scale = int(b[4] / b[3])
            if crop_scale > 10:
                nik_digits_scale = crop_scale

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_mask = auto_canny(blurred)
        img_dilation = cv2.dilate(edge_mask, kernel, iterations=1)
        cv2.imwrite("./tmp/Edges"+str(i)+".jpg", edge_mask)
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

