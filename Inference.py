from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from huggingface_hub import login
import torch
import time

class Inference:
    
    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": self.prompt,
            },
        ]
    
    def __init__(self,load_in_4bit: bool, load_in_8bit: bool,huggingface_token: str, prompt: str = "You are a helpful assistant that answer questions",model_name: str = "02loveslollipop/Furina-2_6-phi-2",device: str = "cuda",torch_dtype: str = "auto",use_flash_attention_2: bool = False) -> None:
        compute_dtype = getattr(torch, "float16")
        
        if use_flash_attention_2:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None
        
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                attn_implementation=attn_implementation,
                
            )
            self.load_in_4bit = True
            self.load_in_8bit = False
        
        if load_in_8bit and not load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type='nf4',
                bnb_8bit_compute_dtype=compute_dtype,
                bnb_8bit_use_double_quant=True,
                attn_implementation=attn_implementation,
            )
            self.load_in_4bit = False
            self.load_in_8bit = True
            
        if not load_in_4bit and not load_in_8bit:
            bnb_config = BitsAndBytesConfig()
            
        if load_in_4bit and load_in_8bit:
            raise UserWarning("Both load_in_4bit and load_in_8bit are set to True, Using 4 bit quantization")
        
        torch.set_default_device(device)
        login(token=huggingface_token)
        peft = PeftConfig.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch_dtype,quantization_config=bnb_config)
        self.model = AutoModelForCausalLM.from_pretrained(peft.base_model_name_or_path,trust_remote_code=True, torch_dtype=torch_dtype,quantization_config=bnb_config,attn_implementation=attn_implementation)
        self.tokenizer = AutoTokenizer.from_pretrained(peft.base_model_name_or_path, trust_remote_code=True,quantization_config=bnb_config)
        self.model = PeftModel(self.model, peft)
        
        self.messages = []
        self. prompt = prompt
        self.reset()
        
    def generate(self, input: str, max_new_tokens: int = 200, do_sample: bool = True, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95) -> tuple[str, float, int]:
        self.messages.append({"role": "user", "content": input})
        prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        begin = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,eos_token_id=self.tokenizer.eos_token_id, forced_eos_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.batch_decode(outputs)[0]
        used_time = time.time() - begin
        length = len(outputs[0]) - len(inputs["input_ids"][0])
        text_generated = text[len(prompt):][:-10]
        self.messages.append({"role": "assistant", "content": text_generated})
        return (text_generated, used_time, length)