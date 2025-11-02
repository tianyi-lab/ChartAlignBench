import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig


class Phi35:
    instance = None
    model = None
    processor = None

    def __init__(self, device="cuda:0"):
        model_id = "microsoft/Phi-3.5-vision-instruct"  
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map= device, trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation='flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
    def infer(self, prompt_text_template, images_list=None):
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                text_to_replace = f"<|image_{idx+1}|>"
                prompt_text = prompt_text.replace(tag_to_search, text_to_replace)

        messages = [{"role": "user", "content": prompt_text}]
        
        # print(messages)
        # print(images_list)

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=images_list if images_list else None, return_tensors="pt").to(self.model.device)

        generation_args = {
            "max_new_tokens": 500,
            # "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

class LlaVa16:
    instance = None
    model = None
    processor = None

    def __init__(self, device = "cuda:0"):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to(device)

    def infer(self, prompt_text_template, images_list=None):
        
        # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                prompt_text = prompt_text.replace(tag_to_search, "")
        
        conversation = None
        if not images_list:
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    ],
                },
            ]
        elif len(images_list) == 1:
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                    ],
                },
            ]
        else:
            raise Exception("Currently not coded for multi-image processing")

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = None
        if images_list:
            inputs = self.processor(images=images_list, text=prompt, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # remove input tokens ([INST] <input text> [/INST] ``` <output text> ```)
        input_text_end_idx = response.find("[/INST]") + len("[/INST]")
        response = response[input_text_end_idx:].replace("```", "")
        
        return response


    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

class InternVL25:    
    instance = None
    pipe = None

    def __init__(self, device = "cuda:0", size = 8):
        if size not in [1, 2, 4, 8]:
            raise Exception(f"size: {size}B for Intern-VL-2.5 not available")
        model_id = f'OpenGVLab/InternVL2_5-{size}B'
        print(F"MODEL IS: {model_id}")
        # integer value, -1 for CPU | >=0 for corresponding GPU no
        device_int = int(device.replace("cuda:", "")) if "cuda" in device else int(-1)
        self.pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192), device = device_int)

    def __init__(self, device = "cuda:0"):
        model_id = 'OpenGVLab/InternVL2-8B'
        # integer value, -1 for CPU | >=0 for corresponding GPU no
        device_int = int(device.replace("cuda:", "")) if "cuda" in device else int(-1)
        self.pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192), device = device_int)

    def infer(self, prompt_text_template, images_list=None):
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                prompt_text = prompt_text.replace(tag_to_search, "<IMAGE_TOKEN>")

        gen_config = GenerationConfig(temperature = 0.0)    # want greedy decoding (do_sample arg not there, checked 'help')

        response = None
        if images_list:
            response = self.pipe((prompt_text, images_list), gen_config=gen_config)
        else:
            response = self.pipe(prompt_text, gen_config=gen_config)
        
        return response.text

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance
