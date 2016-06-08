
class Prior(object):

    def sample(self, num_samples):
        raise NotImplementedError()
        
    def get_name(self):
        raise NotImplementedError()
