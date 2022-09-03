class EnvTest(object):
    def __init__(self):
        self.lagrange_factor = None
        self.state_ = 1
        self.count = 0
        self.state_number = 100
        self.state = [self.state_, self.count]
        self.done = False
        self.penalty = 0.0

    def reset(self):
        self.state_ = 1
        self.count = 0
        self.state = [self.state_, self.count]
        self.done = False
        self.penalty = 0.0
        self.lagrange_factor = None

    def step(self, action):
        assert action in [0, 1], 'action limit!'
        self.done = bool(self.state_ >= self.state_number)
        self.state_ += 1
        self.count += action
        reward = action
        if self.done:
            reward = action - self.count * self.lagrange_factor
            self.penalty = self.count * self.lagrange_factor
        self.state = [self.state_, self.count]
        return self.state, reward, self.done
