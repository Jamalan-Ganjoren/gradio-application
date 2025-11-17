import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

# Text generation function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=4, label="Enter your prompt here"),
    outputs=gr.Textbox(lines=8, label="Falcon Output"),
    title="Falcon-RW-1B Text Generator",
    description="A small open-weight LLM (tiiuae/falcon-rw-1b) running locally — no API required."
)

if __name__ == "__main__":
    demo.launch()
