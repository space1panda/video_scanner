from ultralytics import YOLO
from gradio_webrtc import WebRTC
import gradio as gr
import torch

model = YOLO('best.pt')


@torch.no_grad
def detection(frame):
    res = model(frame[..., ::-1])
    frame = res[0].plot()
    return frame[..., ::-1]


css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""


with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        YOLOv10 Webcam Stream (Powered by WebRTC)
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream")
        image.stream(
            fn=detection, inputs=[image], outputs=[image])


if __name__ == "__main__":
    demo.launch(share=False)
