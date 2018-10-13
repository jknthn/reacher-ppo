import json


class ExperimentManager(object):

    def __init__(self, id):
        self.path = f'ex{id}.txt'
        self.dict = {}

        content = json.dumps(self.dict)
        print(content)
        with open(self.path, 'w+') as f:
            f = content





e = ExperimentManager({
    'TAU': [0.999]
}, 1)