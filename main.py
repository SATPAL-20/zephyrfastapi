from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig

import os

app = FastAPI()

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: List[str]

class UserInput(BaseModel):
    user_prompt: str

def format_prompt(user_prompt: str):
    return f"### Instruction:\n{user_prompt}\n\n### Response:"

def generate(llm: AutoModelForCausalLM, generation_config: GenerationConfig, user_prompt: str):
    """Run model inference, will return a Generator if streaming is true"""
    return llm.generate(
        format_prompt(user_prompt),
        **asdict(generation_config),
    )

config = AutoConfig.from_pretrained(
    os.path.abspath("models"),
    context_length=2048,
)

llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath("models/zephyr.gguf"),
    model_type="Mistral",
    config=config,
)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,
    seed=42,
    reset=True,
    stream=True,
    threads=int(os.cpu_count() / 6),
    stop=[""],
)

user_prefix = "[user]: "
assistant_prefix = "[assistant]:"

@app.post("/generate")
async def generate(user_prompt: str):
    # Load the model
    config = AutoConfig.from_pretrained("models", context_length=2048)
    llm = AutoModelForCausalLM.from_pretrained("models/zephyr.gguf", model_type="Mistral", config=config)

    # Define the generation configuration
    generation_config = {
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "seed": 42,
        "reset": True,
        "stream": True,
        "threads": int(os.cpu_count() / 6),
        "stop": [""]
    }

    # Format the prompt
    formatted_prompt = f"""### Instruction:
{user_prompt}

### Response:"""

    # Generate text
    generated_text = llm(formatted_prompt, **generation_config)

    return {
        "text": generated_text
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000) use something else instead of os library
