from random import sample


class ExperienceReplay:
    '''
    Experience replay buffer
    '''
    def __init__(self, buffer_size=256, min_exp=64):
        '''
        buffer_size: Max number of experiences recorded. If overflow, oldest experience will be discarded
        min_exp: minimum number of experiences required to start sampling
        '''
        assert buffer_size >= min_exp, 'Size of buffer must be greater than or equal to min_exp'
        self.buffer_size = buffer_size
        self.min_exp = min_exp
        self.buffer = list()
        self.__i = 0

    def push(self, s, a, sp, r):
        '''
        Push new experience to buffer
        s: state
        a: action
        sp: next state
        r: reward
        '''
        new_item = (s, a, sp, r)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(new_item)
        else:
            self.buffer[self.__i] = new_item
        # Moving index forward
        self.__i = (self.__i + 1) % self.buffer_size

    def sample(self, size):
        '''
        Get samples from experiences. If buffer have not enough experiences, return empty list
        size: Number of samples
        return list of <size> samples
        '''
        if len(self.buffer) < self.min_exp:
            return list()
        return sample(self.buffer, size)