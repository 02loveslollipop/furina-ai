from Inference import Inference
from confLoader import ConfLoader
import time
import pandas as pd

if __name__ == "__main__":
    conf = ConfLoader()
    
    #load prompt.txt as prompt
    
    try:
        prompt = open("prompt.txt", "r").read()
    except FileNotFoundError:
        print("prompt.txt not found")
        prompt = "You are a helpful assistant that answer questions"
    
    ai_model = Inference(load_in_4bit=False, load_in_8bit=True, huggingface_token=conf.token, model_name=conf.model, device=conf.device, torch_dtype="auto",prompt=prompt,use_flash_attention_2=False)
    
    next = True
    
    while next:
        prompt = input("Enter your question: ")
        now = time.time()
        print(ai_model.generate(prompt,max_new_tokens=650))
        print("generated in: ", time.time() - now)
        next = input("Do you want to continue? (y/n): ").lower() == "y"
        if next:
            print(f"current batch: {ai_model.messages}")
            input("Press enter to continue")
        else:
            save = input("Do you want to save the conversation? (y/n): ").lower() == "y"
            if save:
                name = input("Enter the name of the file: (without extension): ")
                df = pd.DataFrame(ai_model.messages)
                saved = False
                while not saved:
                    try:
                        df.to_csv(f"{name}.csv",index=False)
                        print(f"conversation saved as {name}.csv")
                        saved = True
                    except PermissionError:
                        request = input("PermissionError: the file is already open, do you want to try again? (y/n): ")
                        if request.lower() == "n":
                            saved = True
                        
                    
                