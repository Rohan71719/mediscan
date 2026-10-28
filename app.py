import gradio as gr
from PIL import Image
from predict import analyze, load

NOTE = ("Screening aid for children aged 1 to 5. Not a diagnosis. "
        "A normal result does not rule out pneumonia in a child with symptoms.")

def run(f):
    if f is None: return "Upload a chest X-ray image (JPEG or PNG).", None
    path = f if isinstance(f, str) else f.name
    r = analyze(path)
    if r["status"] == "REJECTED":
        txt = f"REJECTED: {r['reason']}.\nNo prediction is made. Upload a frontal chest X-ray of a child."
    else:
        txt = f"{r['status']}\nProbability of pneumonia: {r['prob']:.3f}"
    try: img = Image.open(path).convert("L")
    except Exception: img = None
    return txt + "\n\n" + NOTE, img

load()
demo = gr.Interface(fn=run, inputs=gr.File(label="Chest X-ray image", file_types=["image"], type="filepath"),
                    outputs=[gr.Textbox(label="Result", lines=5), gr.Image(label="Image received")],
                    title="MediScan (ArogyaNet)", description=NOTE, flagging_mode="never")
if __name__ == "__main__": demo.launch(share=True)
