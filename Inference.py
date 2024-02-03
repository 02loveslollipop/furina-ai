from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from huggingface_hub import login
import torch
import datetime as dt
from confLoader import ConfLoader

class Inference:
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request as Furina.

### Instruction:
{}
        
"""

    prompt = """
### Input:
{}

### Response:
        
""" 
    
    def reset(self,prompt: str = None) -> None:
        
        if prompt is None:
            self.batch = self.alpaca_prompt.format(self.prompt)
        else:
            self.batch = self.alpaca_prompt.format(prompt)
    
    def __init__(self, model_name: str,huggingface_token: str ,prompt: str, device: str = "cpu") -> None:
        
        login(token=huggingface_token)
        self.prompt = prompt
        self.device = device
        self.reset()
        config = PeftConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model,model_name)
    
    def generate(self, text: str, max_new_tokens: int =200, repetition_penalty: float =1.2) -> str:
                
        self.batch = self.batch + self.prompt.format(text)
        inputs = self.tokenizer(self.batch, return_tensors='pt').to("cuda")
        begin = dt.datetime.now()
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty)
        self.batch = f"{self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\n"
        print(f"{len(outputs[0])} tokens generated in {dt.datetime.now()-begin} seconds.")
        return self.batch
        
class CPUInference(Inference):
    def __init__(self, model_name: str,huggingface_token: str ,prompt: str):
        super().__init__(model_name,huggingface_token,prompt,device="cpu")
        self.device = "cpu"
        
    def generate(self, text, max_new_tokens=200, repetition_penalty=1.2):
        self.batch = self.batch + self.prompt.format(text)
        inputs = self.tokenizer(self.batch, return_tensors='pt').to("cpu")
        begin = dt.datetime.now()
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty)
        self.batch = f"{self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\n"
        print(f"{len(outputs[0])} tokens generated in {dt.datetime.now()-begin} seconds.")
        return self.batch

class CUDAInference(Inference):
    def __init__(self, model_name: str,huggingface_token: str ,prompt: str,device: str = "cuda") -> None:
        super().__init__(model_name,huggingface_token,prompt,device=device)
        self.device = device
        
    def generate(self, text: str, max_new_tokens: int =200, repetition_penalty: float =1.2) -> str:
        self.batch = self.batch + self.prompt.format(text)
        inputs = self.tokenizer(self.batch, return_tensors='pt').to(self.device)
        print(self.batch)
        begin = dt.datetime.now()
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty)
        self.batch = f"{self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\n"
        print(f"{len(outputs[0])} tokens generated in {dt.datetime.now()-begin} seconds.")
        return self.batch
       
        
if __name__ == "__main__":
    
    config = ConfLoader()
    
    with open("prompt.txt", "r") as file:
        prompt = file.read()
        file.close()
    
    model = CUDAInference(config.model,config.token,prompt)
    
    print(model.generate("hello", max_new_tokens=20, repetition_penalty=1.2))

#GPU 44 segundos