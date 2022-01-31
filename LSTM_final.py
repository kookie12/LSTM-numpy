import csv
import numpy as np
import emoji
from emo_utils import *
from matplotlib import pyplot as plt

class data:
    def __init__(self, is_train):
        self.is_train = is_train
        print('init')

    def dataloader(self):
        path_1 = "./data_train_test/" # you have to change it according to your path
        path_2 = "../glove.6B/"
        if self.is_train == True:
            X, Y = read_csv(path_1 + 'train_emoji.csv') # x에 대문자 포함해서 문장 단위가 들어옴
        elif self.is_train == False:
            X, Y = read_csv(path_1 + 'test_emoji.csv')
        Y_onehot = []
        X_split = []
        for i in range(len(X)):
            X_split.append(X[i].lower().split())
            onehot = np.zeros((1, 5))
            onehot[0][Y[i]] = 1
            Y_onehot.append(onehot)
        glove_word_to_index, glove_index_to_word, glove_word_to_array = read_glove_vecs(path_2 + "glove.6B.50d.txt")

        X_input= []

        for i in range(len(X_split)):
            word_vector = []
            for word in X_split[i]:
                off = np.ones((1, 50))
                for i in range(50):
                    off[0][i] = glove_word_to_array[word][i]
                word_vector.append(off)
            X_input.append(word_vector)

        X_input = np.array(X_input) # 1*50 vector
        Y_onehot = np.array(Y_onehot) # 1*5 one hot vector
        
        return X_input, Y_onehot, X_split

class function:
    def Cross_Entropy_Loss(self, softmax_matrix, label_matrix):
        delta = 1e-7 
        return -np.sum(label_matrix*np.log(softmax_matrix+delta))

    def softmax(self, x):
        s = np.exp(x)
        total = np.sum(s, axis=0).reshape(-1,1)
        return s/total

    def tanh(self, x):
        s = (1 - np.exp(-1*x))/(1 + np.exp(-1*x))
        return s

    def differ_softmax_cross(self, softmax_matrix, label_matrix):
        x = (softmax_matrix - label_matrix)
        return x

    def differ_tanh(self, x):
        output = 1 - self.tanh(x)*self.tanh(x)
        return output

    def sigmoid(self, x):
        s = 1/(1 + np.exp(-1*x))
        return s

    def differ_sigmoid(self, x):
        #return self.sigmoid(x)*(1-self.sigmoid(x))
        return x * (1-x)

class LSTM_cell(function):
    def __init__(self, hidden_size, input_size):
        np.random.seed(0)
        self.H = hidden_size  # H
        self.D = input_size   # D
        self.N = 1
        self.Wx = np.random.randn(self.D, self.H * 4) / np.sqrt(self.H) # D * 4H = 50 * 40
        self.Wh = np.random.randn(self.H, self.H * 4) / np.sqrt(self.H) # H * 4H = 10 * 40
        self.b = np.random.randn(self.N, self.H * 4) / np.sqrt(self.H)  # N * 4H = 1 * 40
        self.dWx = np.random.randn(self.D, self.H * 4) / np.sqrt(self.H)
        self.dWh = np.random.randn(self.H, self.H * 4) / np.sqrt(self.H)
        self.db = np.random.randn(self.N, self.H * 4) / np.sqrt(self.H)
        self.moment_Wh_1 = 0
        self.moment_Wh_2 = 0
        self.moment_Wx_1 = 0
        self.moment_Wx_2 = 0
        self.moment_b_1 = 0
        self.moment_b_2 = 0
        self.rho = 0.9
        self.decay_rate = 0.9

    def lstm_cell_forward(self, c_input, h_input, x_input):
        self.c_input = c_input # N * H = 1 * 10
        self.h_input = h_input # N * H = 1 * 10
        self.x_input = x_input # N * D = 1 * 50
        self.A = np.dot(self.x_input, self.Wx) + np.dot(self.h_input, self.Wh) + self.b # 1 * 4H
        
        self.ft = super().sigmoid(self.A[:, :self.H])           # 1 * H
        self.it = super().sigmoid(self.A[:, self.H:2*self.H])   # 1 * H
        self.ctt = super().tanh(self.A[:, 2*self.H:3*self.H])  # 1 * H
        self.ot = super().sigmoid(self.A[:, 3*self.H:4*self.H]) # 1 * H
        
        self.c_next = self.ft * self.c_input + self.it * self.ctt  # 1 * H
        self.h_next = self.ot * super().tanh(self.c_next)          # 1 * H 

    def lstm_cell_backward(self, dc_next, dh_next): # 10 * 1 로 들어옴
        
        temp = dc_next + (dh_next * self.ot) * super().differ_tanh(self.c_next)

        self.dft = temp * self.c_input
        self.dit = temp * self.ctt
        self.dctt = temp * self.it
        self.dot = dh_next * np.tanh(self.c_next)

        # 활성함수 밑으로 전달해주어야 해서 자기자신 속미분
        self.dft *= super().differ_sigmoid(self.ft)
        self.dit *= super().differ_sigmoid(self.it)
        self.dctt *= (1 - self.ctt ** 2)
        self.dot *= super().differ_sigmoid(self.ot)

        # 한번에 계산
        dA = np.hstack((self.dft, self.dit, self.dctt, self.dot))

        self.dWh = np.dot(self.h_input.T, dA) # 1 * 10, 1 * 40
        self.dWx = np.dot(self.x_input.T, dA) # 
        self.db = dA.sum(axis=0)

        self.dx_prev = np.dot(dA, self.Wx.T) # 10 * 40, 40 * 50
        self.dh_prev = np.dot(dA, self.Wh.T) # 10 * 40, 40 * 10
        self.dc_prev = temp * self.ft

    def SGD(self, learning_rate):
        self.Wx -= learning_rate * self.dWx
        self.Wh -= learning_rate * self.dWh
        self.b -= learning_rate * self.db

    def ADAM(self, learning_rate):
        # self.Wx
        self.moment_Wx_1 = self.rho * self.moment_Wx_1 + (1 - self.rho) * self.dWx
        self.moment_Wx_2 = self.decay_rate * self.moment_Wx_2 + (1 - self.decay_rate) * self.dWx * self.dWx
        self.Wx -= learning_rate * self.moment_Wx_1 / (np.sqrt(self.moment_Wx_2) + 1e-7)

        # self.Wh
        self.moment_Wh_1 = self.rho * self.moment_Wh_1 + (1 - self.rho) * self.dWh
        self.moment_Wh_2 = self.decay_rate * self.moment_Wh_2 + (1 - self.decay_rate) * self.dWh * self.dWh
        self.Wh -= learning_rate * self.moment_Wh_1 / (np.sqrt(self.moment_Wh_2) + 1e-7)

        # self.b
        self.moment_b_1 = self.rho * self.moment_b_1 + (1 - self.rho) * self.db
        self.moment_b_2 = self.decay_rate * self.moment_b_2 + (1 - self.decay_rate) * self.db * self.db
        self.b -= learning_rate * self.moment_b_1 / (np.sqrt(self.moment_b_2) + 1e-7)

class LSTM_MODEL(function):
    def __init__(self, X_input, Y_input, X_test_input, Y_test_input, X_split, X_test_split, learning_rate, epoch, optimizer, dropout):
        np.random.seed(0)
        self.X_input = X_input
        self.Y_input = Y_input
        self.X_test_input = X_test_input
        self.Y_test_input = Y_test_input
        self.X_split = X_split
        self.X_test_split = X_test_split
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.optimizer = optimizer
        self.dropout = dropout
        self.hidden_size = 40
        self.input_size = 50
        self.LSTM_1st_layer = [LSTM_cell(hidden_size = self.hidden_size, input_size = self.input_size) for _ in range(10)]
        self.LSTM_2nd_layer = [LSTM_cell(hidden_size = self.hidden_size, input_size = self.hidden_size) for _ in range(10)]
        self.Why = np.random.randn(5, self.LSTM_2nd_layer[0].H) / np.sqrt(self.LSTM_2nd_layer[0].H)
        self.by = np.random.randn(5, 1) / np.sqrt(self.LSTM_2nd_layer[0].H)
        self.c_before = np.zeros((1, self.LSTM_2nd_layer[0].H))
        self.h_before = np.zeros((1, self.LSTM_2nd_layer[0].H))
        self.moment_Why_1 = 0
        self.moment_Why_2 = 0
        self.moment_by_1 = 0 
        self.moment_by_2 = 0
        self.rho = 0.9
        self.decay_rate = 0.9
        self.loss, self.loss_list = [], []
        self.test_loss, self.test_loss_list = [], []
        self.train_accuracy_list, self.test_accuracy_list = [], []
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.train_prediction, self.test_prediction = [], []
        self.answer_emoji = []

    def model_start(self):
        for epoch in range(self.epoch):
            for iteration in range(len(self.X_input)):
                x_inputs = self.X_input[iteration]
                y_inputs = self.Y_input[iteration]

                self.forward(x_inputs, y_inputs, is_train = True)
                self.backward(x_inputs)

            loss = round(sum(self.loss)/len(X_input) ,2)
            train_accuracy = round(self.train_accuracy/len(X_input) * 100, 2)

            if epoch % 10 == 0:
                print("train... ", epoch, " epoch -> loss :", loss, " train accuracy : ", train_accuracy, "%")

            self.loss_list.append(loss) # [0][0]
            self.train_accuracy_list.append(train_accuracy)
            self.loss = [] 
            self.train_accuracy = 0

            # test
            for iteration in range(len(self.X_test_input)):
                x_test_inputs = self.X_test_input[iteration]
                y_test_inputs = self.Y_test_input[iteration]
                self.forward(x_test_inputs, y_test_inputs, is_train = False)

            loss = round(sum(self.test_loss)/len(X_test_input), 2)
            test_accuracy = round(self.test_accuracy/len(X_test_input) * 100, 2)

            if epoch % 10 == 0:
                print("test ... ", epoch, " epoch -> loss :", loss, " test accuracy : ", test_accuracy, "%")

            self.test_loss_list.append(round((sum(self.test_loss)/len(X_test_input)), 2))
            self.test_accuracy_list.append(test_accuracy)
            self.test_loss = []
            self.test_accuracy = 0

            # print emoji
            if epoch == self.epoch - 1:
                for i, items in enumerate(self.X_test_split):
                    temp = str(i) + ". "
                    prediction = self.test_prediction[i]
                    answer = self.answer_emoji[i]
                    for item in items:
                        temp += item + " "
                    temp += "->"
                    print(temp, " test : ", label_to_emoji(prediction)) # "answer : ", label_to_emoji(answer), 

            self.test_prediction = []
            self.train_prediction = []
            self.answer_emoji = []

            # train & test loss graph
            if epoch  == self.epoch - 1:
                plt.subplot(2, 2, 1)
                plt.title('LSTM + SGD + 50d loss with dropout', fontsize=12)
                plt.plot(range(0, epoch+1), self.loss_list, 'b', label = 'train')
                plt.plot(range(0, epoch+1), self.test_loss_list, 'r', label = 'test')
                plt.ylabel('Cost')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')

                plt.subplot(2, 2, 2)
                plt.title('LSTM + SGD + 50d accuracy with dropout', fontsize=12) #  with dropout
                plt.plot(range(0, epoch+1), self.train_accuracy_list, 'b', label = 'train')
                plt.plot(range(0, epoch+1), self.test_accuracy_list, 'r', label = 'test')
                plt.ylabel('accuracy (%)')
                plt.xlabel('Epochs')
                plt.legend(loc='upper right')
                plt.show()

    def forward(self, x_inputs, y_inputs, is_train):
        h_before_1, h_before_2 = self.h_before, self.h_before # 맨 처음엔 0
        c_before_1, c_before_2 = self.c_before, self.c_before
        y_input = y_inputs.reshape(5, 1)
        self.ht, self.diff_ht = [], []
        self.ct, self.diff_ct = [], []
        self.ht_2, self.diff_ht_2 = [], []
        self.ct_2, self.diff_ct_2 = [], []
        
        for n, x_input in enumerate(x_inputs):
            x_input = x_input.reshape(1,50)
            self.LSTM_1st_layer[n].lstm_cell_forward(c_before_1, h_before_1, x_input)
            c_before_1 = self.LSTM_1st_layer[n].c_next
            h_before_1 = self.LSTM_1st_layer[n].h_next
            self.ct.append(c_before_1)

            # dropout layer
            if self.dropout == "yes":
                h_before_1 = self.dropout_layer(h_before_1, 0.5)
                self.ht.append(h_before_1)

            else:
                self.ht.append(h_before_1)

            self.LSTM_2nd_layer[n].lstm_cell_forward(c_before_2, h_before_2, h_before_1) # self.ht[n]
            c_before_2 = self.LSTM_2nd_layer[n].c_next
            h_before_2 = self.LSTM_2nd_layer[n].h_next
            self.ct_2.append(c_before_2)
            self.ht_2.append(h_before_2)

            if n == len(x_inputs) - 1:
                y_predic = np.dot(self.Why, self.ht_2[-1].T) + self.by # 5*20, 20*1 = 5*1
                y_softmax = super().softmax(y_predic)
                loss = super().Cross_Entropy_Loss(y_softmax, y_input)
                self.diff_softmax_cross = super().differ_softmax_cross(y_softmax, y_input)

        if is_train == True:
            prediction = np.argmax(y_softmax)
            answer = np.argmax(y_input)
            if prediction == answer:
                self.train_accuracy += 1
            self.loss.append(loss)

        elif is_train == False:
            prediction = np.argmax(y_softmax)
            answer = np.argmax(y_input)
            if prediction == answer:
                self.test_accuracy += 1
            self.test_loss.append(loss)
            self.test_prediction.append(prediction)
            self.answer_emoji.append(answer)

    def dropout_layer(self, x, prob):
        u = np.random.rand(*x.shape) < prob
        x *= u
        return x

    def backward(self, x_inputs):
        self.last_dht_2 = (np.dot(self.Why.T, self.diff_softmax_cross)).T # 2층에서 2층으로 전달하는 ht
        self.last_dct_2 = np.zeros((1, self.hidden_size))
        self.last_dht_1 = np.zeros((1, self.hidden_size)) # 1층에서 1층으로 전달하는 ht
        self.last_dct_1 = np.zeros((1, self.hidden_size))
        self.last_dx_1 = np.zeros((1, self.hidden_size)) # 2층에서 1층으로 내려가는 x

        for n in reversed(range(0, len(x_inputs))):
            self.LSTM_2nd_layer[n].lstm_cell_backward(self.last_dct_2, self.last_dht_2) # self.diff_ht_2[n]
            self.last_dht_2 = self.LSTM_2nd_layer[n].dh_prev
            self.last_dct_2 = self.LSTM_2nd_layer[n].dc_prev
            self.last_dx_1 = self.LSTM_2nd_layer[n].dx_prev
            self.last_dht_1 += self.last_dx_1
            self.LSTM_1st_layer[n].lstm_cell_backward(self.last_dct_1, self.last_dht_1) # self.diff_ht[n]
            self.last_dht_1 = self.LSTM_1st_layer[n].dh_prev
            self.last_dct_1 = self.LSTM_1st_layer[n].dc_prev

        for m in range(0, len(x_inputs)):
            if self.optimizer == "SGD":
                self.LSTM_1st_layer[m].SGD(self.learning_rate)
                self.LSTM_2nd_layer[m].SGD(self.learning_rate)
            else:
                self.LSTM_1st_layer[m].ADAM(self.learning_rate)
                self.LSTM_2nd_layer[m].ADAM(self.learning_rate)

        self.dWhy = np.dot(self.diff_softmax_cross, self.ht_2[-1])
        self.dby = self.diff_softmax_cross

        if self.optimizer == "SGD":
            self.Why -= self.learning_rate * self.dWhy
            self.by -= self.learning_rate * self.dby
        else: 
            # self.Why
            self.moment_Why_1 = self.rho * self.moment_Why_1 + (1 - self.rho) * self.dWhy
            self.moment_Why_2 = self.decay_rate * self.moment_Why_2 + (1 - self.decay_rate) * self.dWhy * self.dWhy
            self.Why -= self.learning_rate * self.moment_Why_1 / (np.sqrt(self.moment_Why_2) + 1e-7)

            # self.by
            self.moment_by_1 = self.rho * self.moment_by_1 + (1 - self.rho) * self.dby
            self.moment_by_2 = self.decay_rate * self.moment_by_2 + (1 - self.decay_rate) * self.dby * self.dby
            self.by -= self.learning_rate * self.moment_by_1 / (np.sqrt(self.moment_by_2) + 1e-7)
        
if __name__ == "__main__":
    _data = data(is_train = True)
    _test_data = data(is_train = False)
    X_input, Y_input, X_split = _data.dataloader()
    X_test_input, Y_test_input, X_test_split = _test_data.dataloader()
    lstm = LSTM_MODEL(X_input, Y_input, X_test_input, Y_test_input, X_split, X_test_split, 0.001, 2000, optimizer = "SGD", dropout = "yes") # without dropout 0.03, 400, 5, 50 / with dropout 
    lstm.model_start()