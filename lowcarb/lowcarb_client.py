class LowcarbClient:
    def __init__(self):
        self.location = None

    def get_properties(self, config):
        return {'location': self.location}


def lowcarb(func):
    def wrapper(*args):
        properties = func(*args)
        properties['location'] = args[0].location

        return properties

    return wrapper
