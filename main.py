from Inference import Inference
from confLoader import ConfLoader
import time

if __name__ == "__main__":
    conf = ConfLoader()
    
    ai_model = Inference(load_in_4bit=False, load_in_8bit=True, huggingface_token=conf.token, model_name=conf.model, device=conf.device, torch_dtype="auto")
    
    next = True
    
    while next:
        prompt = input("Enter your question: ")
        now = time.time()
        print(ai_model.generate(prompt))
        print("generated in: ", time.time() - now)
        next = input("Do you want to continue? (y/n): ").lower() == "y"