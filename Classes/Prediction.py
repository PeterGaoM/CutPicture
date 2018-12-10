import tensorflow as tf
from tools.util import *


class Prediction():
    def __init__(self, model_path, input_name, output_name):
        self.graph = self.load_graph(model_path)
        self.input = self.graph.get_tensor_by_name('prefix/'+input_name)
        if output_name is list:
            for name in output_name:
                self.ouput = []
                self.ouput.append('prefix/'+name)
        else:
            self.ouput = self.graph.get_tensor_by_name('prefix/'+output_name)
        self.sess = tf.Session(graph=self.graph)  # 创建新的sess

    def load_graph(self, frozen_graph_file):
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
        return graph

    def predict(self, img, RESIZED_IMAGE = (100,100)):
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, RESIZED_IMAGE)

        img = img[np.newaxis, :, :, np.newaxis]
        return self.sess.run(self.ouput, feed_dict={self.input: img})


class crossdetect():
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model_path + '.meta')

        self.sess = tf.Session(graph=self.graph)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model_path)  # 从恢复点恢复参数

    def predict(self, img):
        height, width = img.shape[:2]
        s_h = np.ones(1) * height / 300
        s_w = np.ones(1) * width / 300
        img = cv2.resize(img, (300, 300))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = np.reshape(img, [1, 300, 300, 3])
        cls_predict = self.graph.get_tensor_by_name('output:0')
        reg_predict = self.graph.get_tensor_by_name('ssd_ext/strided_slice_1:0')
        ssd_input = self.graph.get_tensor_by_name('input:0')
        cls_scores, reg_offsets = self.sess.run([cls_predict, reg_predict], feed_dict={ssd_input: img})
        det_boxes = post_process(gen_anchors(), cls_scores, reg_offsets, s_w, s_h, score_thr=0.98, nms_thr=0.3)

        return det_boxes








