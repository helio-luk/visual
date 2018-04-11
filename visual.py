import numpy as np
import matplotlib as mp
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import warnings
import matplotlib.cbook
import cv2
from data import Data
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

logs_path = "/tmp/mnist/2"

#modelo
def getInit():
    shape = 50
    #data
    x         = tf.placeholder(tf.float32, [None, shape*shape], name="x-in")
    true_y    = tf.placeholder(tf.float32, [None, 13], name="y-in")
    keep_prob = tf.placeholder("float")

    #model
    x_image  = tf.reshape(x,[-1,shape,shape,1])
    hidden_1 = slim.conv2d(x_image,5,[5,5])
    pool_1   = slim.max_pool2d(hidden_1,[2,2])
    hidden_2 = slim.conv2d(pool_1,5,[5,5])
    pool_2   = slim.max_pool2d(hidden_2,[2,2])
    hidden_3 = slim.conv2d(pool_2,20,[5,5])
    hidden_3 = slim.dropout(hidden_3,keep_prob)
    out_y    = slim.fully_connected(slim.flatten(hidden_3),13,activation_fn=tf.nn.softmax)

    cross_entropy      = -tf.reduce_sum(true_y*tf.log(out_y))
    tf.summary.scalar("cross_entropy", cross_entropy)

    correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)

    train_step         = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    batchSize = 50
    summary_op = tf.summary.merge_all()

    init      = tf.global_variables_initializer()

    return init, train_step, true_y, x, keep_prob, accuracy, hidden_1, hidden_2, hidden_3, summary_op

#imagem de teste normalizada
def getOriginal(imageToUse):
    shape = 50
    aux_norm=cv2.normalize(imageToUse,0,255,cv2.NORM_L1)
    aux_norm = np.reshape(aux_norm, [shape,shape])
    aux_norm_c = aux_norm.astype(np.uint8)
    aux_norm_c = cv2.normalize(aux_norm_c,0,255,cv2.NORM_L1)
    aux_norm_c = cv2.resize(aux_norm_c, (200,200), interpolation=cv2.INTER_CUBIC)
    aux_norm_c = aux_norm_c*5

    return aux_norm_c

#imagem normalizada apos camadas
def getNorm(aux_norm_j):
    aux_reshape_c = aux_norm_j.astype(np.uint8)

    aux_reshape_c = cv2.normalize(aux_reshape_c,0,255,cv2.NORM_L1)
    aux_reshape_c = cv2.resize(aux_reshape_c, (200,200), interpolation=cv2.INTER_CUBIC)
    aux_reshape_c = cv2.equalizeHist(aux_reshape_c)
    return aux_reshape_c

'''
def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,50*50],order='F'),keep_prob:1.0})
    plotNNFilter(units)


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1

    for i in range(filters):
        #print units[0,:,:,i]

        j=cv2.normalize(units[0,:,:,i],0,255,cv2.NORM_L1)

        if(j.shape[0] == 50):

            k = np.reshape(j, [50,50])
            c = k.astype(np.uint8)
            c = cv2.normalize(c,0,255,cv2.NORM_L1)
            c = cv2.resize(c, (200,200), interpolation=cv2.INTER_CUBIC)
            c = c*5
            h = str(i) + '.png'
            cv2.imwrite(h, c)


        elif(j.shape[0] == 25):
            k = np.reshape(j, [25,25])
            c = k.astype(np.uint8)
            c = cv2.normalize(c,0,255,cv2.NORM_L1)
            c = cv2.resize(c, (200,200), interpolation=cv2.INTER_CUBIC)
            c = c*5
            h = str(i) + '25.png'
            cv2.imwrite(h, c)


        else:
            k = np.reshape(j, [12,12])
            c = k.astype(np.uint8)
            c = cv2.normalize(c,0,255,cv2.NORM_L1)
            c = cv2.resize(c, (200,200), interpolation=cv2.INTER_CUBIC)
            c = c*4
            h = str(i) + '.png'
            cv2.imwrite(h, c)
'''

#Training data
data = Data('../videos/train/013009_A29_Block6_C57ma1_t.avi','../videos/train/013009_A29_Block6_C57ma1_t.txt')
train_data = data.getVideoMatrix()
train_labels = data.getLabels()
#Testing data
data.setArquivoVideo('../videos/test/012609_A29_Block1_C57fe1_t.avi')
data.setArquivoLabel('../videos/test/012609_A29_Block1_C57fe1_t.txt')
eval_data = data.getVideoMatrix()
eval_labels = data.getLabels()


tf.reset_default_graph()
sess = tf.Session()
init, train_step, true_y, x, keep_prob, accuracy, hidden_1, hidden_2, hidden_3, sumary = getInit()
sess.run(init)
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

for ep in range(200):
    j = 0
    k = 1

    for i in range(10000):#1001
        if(k+50 <= train_data.shape[0]): #para nao estourar o tamanho da data

            _, summary =sess.run([train_step,sumary], feed_dict={x:train_data[j+1:k+1,:],true_y:train_labels[j+1:k+1,:], keep_prob:0.5})
            writer.add_summary(summary, ep * 100 + i)
            trainAccuracy = sess.run(accuracy, feed_dict={x:train_data[j+1:k+1,:],true_y:train_labels[j+1:k+1,:], keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, trainAccuracy))
            '''
            #Geracao da imagem original para teste
            imageToUse = eval_data[i]
            original = getOriginal(imageToUse)
            cv2.imwrite('original'+str(i)+'.png', original)

            #inicio do teste
            units = sess.run(hidden_1,feed_dict={x:np.reshape(imageToUse,[1,50*50],order='F'),keep_prob:1.0})
            filters = units.shape[3]
            n_columns = 6
            n_rows = math.ceil(filters / n_columns) + 1
            for stepA in range(filters):
                aux_norm_j = cv2.normalize(units[0,:,:,stepA],0,255,cv2.NORM_L1)
                if(aux_norm_j.shape[0] == 50):
                    #breshape = np.reshape(aux_norm_j, [25,25])
                    aux_reshape_h = str(i) + str(stepA) + '.png'
                    cv2.imwrite(aux_reshape_h, getNorm(aux_norm_j))
            '''
            j = k
            k = k+50

testAccuracy = sess.run(accuracy, feed_dict={x:eval_data,true_y:eval_labels, keep_prob:1.0})
print("test accuracy %g"%(testAccuracy))
