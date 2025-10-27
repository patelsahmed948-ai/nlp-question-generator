from fastapi import FastAPI, UploadFile, File
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf8")
    input_text = "generate questions: " + content
    encoding = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(encoding, max_length=64, num_beams=4)
    q = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"questions": q}
