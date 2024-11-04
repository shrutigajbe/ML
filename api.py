
from auth_token import auth_token
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
# Change torch_dtype to float32 for CPU
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float32, use_auth_token=auth_token)
pipe.to(device)

@app.get("/")
def generate(prompt: str): 
    # Remove autocast for CPU
    image = pipe(prompt, guidance_scale=8.5).images[0]
    
    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
