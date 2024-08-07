def get_dataset_config_class(dataset_name):
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class Boiler:
    def __init__(self):
        super(Boiler, self).__init__()
        self.data_base_path = './datasets/Boiler'
        self.scenarios = [("1", "2"), ("1", "3"), ("2", "1"), ("2", "3"), ("3", "1"), ('3', '2')]
        self.input_dim = 15 # 感觉是20。tensorflow 版本的代码是20
        self.class_num = 2
        self.window_size = 6
        self.time_interval = 1
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)

class Air:
    def __init__(self):
        super(Air, self).__init__()
        self.data_base_path = './datasets/Air'
        self.scenarios = [("B", "T"), 
                          ("G", "T"), 
                          ("S", "T"),
                          ]
        self.input_dim = 11
        self.class_num = 1
        self.window_size = 6
        self.time_interval = 1
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)

class Building:
    def __init__(self):
        super(Building, self).__init__()
        self.data_base_path = './datasets/Building'
        self.scenarios = [("Cody", "Anastasia"),
                        #   ("Anastasia", "Cody"), ("Asher", "Cody"),("Asher", "Anastasia"),
                          ]
        self.input_dim = 48
        self.class_num = 1
        self.window_size = 6
        self.time_interval = 1
        self.segments_length = list(range(self.time_interval, self.window_size + 1, self.time_interval))
        self.segments_num = len(self.segments_length)