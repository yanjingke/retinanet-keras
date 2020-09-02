# import tensorflow as tf
# import numpy as np
# sess=tf.Session()
# a=np.array([[1,0,0],[0,1,1]])
# a1=np.array([[3,2,3],[4,5,6]])
# print(sess.run(tf.equal(a,1)))
# print(sess.run(tf.where(tf.equal(a,1),a,a1)))


import tensorflow as tf
from PIL import Image
import numpy as np
img = np.array(Image.open("img/street.jpg"))
shape = img.shape
img = img.reshape([1,shape[0], shape[1], shape[2]])
a = tf.image.crop_and_resize(img,[[0.5,0.6,0.9,0.8],[0.2,0.6,1.3,0.9]],box_ind=[0,0],crop_size=(100,100))
sess = tf.Session()
b = a.eval(session = sess)
Image.fromarray(np.uint8(b[1])).show()