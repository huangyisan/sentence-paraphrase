import gradio as gr
from paraphrase.exec import Pegasus
 

if __name__ == '__main__':
    P = Pegasus(num_return_sequences=2, num_beams=2)
    P.clean()
    demo = gr.Interface(fn=P.exec, inputs="text", outputs="text")
    demo.launch()

