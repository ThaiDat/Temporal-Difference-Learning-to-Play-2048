from globalconfig import gconfig
from random import sample


class ExperienceReplay:
    '''
    Experience replay buffer
    '''
    def __init__(self, buffer_size=gconfig['EXPERIENCE_BUFFER'], min_exp=gconfig['MIN_EXPERIENCE']):
        '''
        buffer_size: Max number of experiences recorded. If overflow, oldest experience will be discarded
        min_exp: minimum number of experiences required to start sampling
        '''
        assert buffer_size >= min_exp, 'Size of buffer must be greater than or equal to min_exp'
        self.buffer_size = buffer_size
        self.min_exp = min_exp
        self.buffer = list()
        self.__i = 0

    def __len__(self):
        '''
        Return the size of buffer
        '''
        return len(self.buffer)

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
        return state bach, action batch, next state batch, reward batch
        '''
        if len(self.buffer) < self.min_exp:
            return list()
        samples = sample(self.buffer, size)
        states = []; actions = []; rewards = []; nxt_states = []
        for s, a, r, sp in samples:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            nxt_states.append(sp)
        return states, actions, rewards, nxt_states