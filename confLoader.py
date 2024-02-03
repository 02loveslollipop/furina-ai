import yaml

class ConfLoader:
    def __init__(self):
    
        try:
            config_file = open('config.yml')
            print("Using config.yml")
        except FileNotFoundError:
            print("You don't have a config.yml file, using example_config.yml")
            config_file = open('example_config.yml')
        finally:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                self.token = config['token']
                self.model = config['model']
                self.device = config['device']
            except KeyError:
                print("Your config file is missing a required field.")
                exit(1)
