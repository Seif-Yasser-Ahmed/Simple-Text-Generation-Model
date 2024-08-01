import gradio as gr
from transformers import pipeline

model = pipeline("text-generation", model="gpt2", tokenizer="gpt2")


def generate_text(input_text):
    return model(input_text)[0]['generated_text']


def create_interface():
    interface = gr.Interface(
        fn=generate_text,
        inputs=[gr.Textbox(label="Enter Prompt", lines=2,
                           placeholder="Type your prompt here...")],
        outputs=[gr.Textbox(label="Generated Text")],
        title="Text Generation with GPT-2",
        description="Enter a prompt and get generated text using GPT-2 model."
    )
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
