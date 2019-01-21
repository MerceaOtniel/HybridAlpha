import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1

        x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same",kernel_regularizer=l2(1e-4))(x_image)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        for i in range(19):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, strides=1,kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        self.policy_out = Dense(self.action_size, activation="softmax",kernel_regularizer=l2(1e-4))(x)

        # for value output
        x = Conv2D(filters=1, kernel_size=1, strides=1,kernel_regularizer=l2(1e-4))(res_out)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu",kernel_regularizer=l2(1e-4))(x)
        self.value_out = Dense(1, activation="tanh",kernel_regularizer=l2(1e-4))(x)

        self.model = Model(inputs=self.input_boards, outputs=[self.policy_out, self.value_out])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=SGD(lr=1e-3,momentum=0.9))

    def _build_residual_block(self, x, index):
        in_x = x
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same",kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same",kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=3)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x