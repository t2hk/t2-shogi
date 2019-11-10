from chainer import Chain
import chainer.functions as F
import chainer.links as L
import copy

from t2engine.common import *

ch = 192
fcl = 256

class Block(Chain):

    def __init__(self, is_wobn):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.is_wobn = is_wobn

    def __call__(self, x):
        #h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.conv1(x))
        h2_wobn = self.conv2(h1)
        h2_bn = self.bn2(h2_wobn)

        result_wobn = F.relu(x + h2_wobn)
        result_bn = F.relu(x + h2_bn)
        #return result_wobn, result_bn
        if(self.is_wobn):
            return result_wobn
        else:
            return result_bn

class T2Resnet_multi(Chain):
    def __init__(self, blocks = 5):
        super(T2Resnet_multi, self).__init__()
        self.blocks = blocks
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            for i in range(1, blocks):
                block_wobn = Block(True)
                block_bn = Block(False)
                self.add_link('b_wobn{}'.format(i), block_wobn)
                self.add_link('b_bn{}'.format(i), block_bn)
            # policy network
            self.policy=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.policy_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            # value network
            self.value1=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.value1_bn = L.BatchNormalization(MOVE_DIRECTION_LABEL_NUM)
            self.value2=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.value3=L.Linear(fcl, 1)

    def __call__(self, x):
        h_wobn = F.relu(self.l1(x))
        h_bn = copy.deepcopy(h_wobn)
        for i in range(1, self.blocks):
            h_wobn = self['b_wobn{}'.format(i)](h_wobn)
            h_bn = self['b_bn{}'.format(i)](h_bn)
        # policy network
        h_wobn_policy = self.policy(h_wobn)
        h_bn_policy = self.policy(h_bn)
        u_wobn_policy = self.policy_bias(F.reshape(h_wobn_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        u_bn_policy = self.policy_bias(F.reshape(h_bn_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))

        u_wobn_policy = self.policy_bias(F.reshape(h_wobn_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        u_bn_policy = self.policy_bias(F.reshape(h_bn_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        #h_value = F.relu(self.value1_bn(self.value1(h)))
        h_wobn_value = F.relu(self.value1(h_wobn))
        h_wobn_value = F.relu(self.value2(h_wobn_value))
        u_wobn_value = self.value3(h_wobn_value)

        h_bn_value = F.relu(self.value1(h_bn))
        h_bn_value = F.relu(self.value2(h_bn_value))
        u_bn_value = self.value3(h_bn_value)

        return u_wobn_policy, u_wobn_value, u_bn_policy, u_bn_value
