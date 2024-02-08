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
                self.token = config['model']['token']
                self.model = config['model']['model']
                self.device = config['model']['device']
                self.max_new_tokens = config['model']['max_new_tokens']
                self.temperature = config['model']['temperature']
                self.top_k = config['model']['top_k']
                self.top_p = config['model']['top_p']
                self.do_sample = config['model']['do_sample']
                self.torch_dtype = config['model']['torch_dtype']
                self.load_in_4bit = config['quantization']['load_in_4bit']
                self.load_in_8bit = config['quantization']['load_in_8bit']
                self.port = config['api']['port']   
                self.host = config['api']['host']
                self.debug = config['api']['debug']
                self.api_token = config['api']['token']
            except KeyError:
                print("Your config file is missing a required field.")
                exit(1)
