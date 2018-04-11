import cv2
import numpy as np

class Data(object):

    def __init__(self, arquivo_video, arquivo_label, labels = [], image_matrix = []):
        self.arquivo_video = arquivo_video
        self.arquivo_label = arquivo_label
        self.labels = labels
        self.image_matrix = image_matrix


    def setArquivoVideo(self, arquivo):
        self.arquivo_video = arquivo
    def setArquivoLabel(self, label):
        self.arquivo_label = label


    def getLabels(self):
        '''
        Retorna array uint8 com as features de cada frame do video
        '''
        def f(x):
            return {
                'approach': 0,
                'attack': 1,
                'copulation': 2,
                'chase': 3,
                'circle': 4,
                'drink': 5,
                'eat': 6,
                'clean': 7,
                'human': 8,
                'sniff': 9,
                'up': 10,
                'walk_away': 11,
                'other': 12,
            }[x]

        a = open(self.arquivo_label)
        last = a.readlines()
        num_frames = last[-1].split(';')[1]

        self.labels = np.zeros((int(num_frames),13), dtype=np.float32)
        aux = np.zeros(13, dtype=np.float32)

        for line in last:
            ini, fin, label, _ = line.split(';')
            aux[f(label)] = 1

            self.labels[int(ini)-1:int(fin)-1,:] = aux#f(label)

        return self.labels

    def getVideoMatrix(self):
        '''
        Returna uma matriz uint8 onde cada linha representa os frames do video
        '''

        video = cv2.VideoCapture(self.arquivo_video)
        a = open(self.arquivo_label)
        last = a.readlines()
        num_frames = last[-1].split(';')[1]

        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        tam = 50*50#h*w
        self.image_matrix = np.zeros((int(num_frames),tam), dtype=np.float32)#COLOCAR 16

        for i in range(0, int(num_frames)):

            ret, frame = video.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (50,50), interpolation = cv2.INTER_AREA)

            x = np.array(image, dtype=np.uint8)
            p = x.flatten().reshape(1,tam)

            self.image_matrix[i,:] = p[:]

        
        return self.image_matrix
