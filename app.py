import gradio as gr
from transformers import pipeline

# Load the mBART translation model
translation_model = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

def translate_text(source_language, target_language, text_to_translate):
    # Use the mBART model to perform translation
    translation_result = translation_model(text_to_translate, src_lang=source_language, tgt_lang=target_language)
    return translation_result[0]['translation_text']

iface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.inputs.Dropdown(["ar", "en", "fr", "de", "es", "zh", "ja"], label="Select Source Language"),
        gr.inputs.Dropdown(["en", "fr", "de", "es", "zh", "ja", "ar"], label="Select Target Language"),
        gr.inputs.Textbox(label="Enter Text to Translate"),
    ],
    outputs=gr.outputs.Textbox(label="Translation Result:")
)

if __name__ == "__main__":
    iface.launch()
