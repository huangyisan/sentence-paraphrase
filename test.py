import gradio as gr
import random


def greet_number(name):
    greeting = "Hello " + name
    lucky_number = random.randint(1, 100)
    return greeting, lucky_number

demo = gr.Interface(fn=greet_number,
             inputs=gr.Textbox(),
             outputs=[gr.Textbox(), gr.Textbox()], #<-- Multiple output components passed as list of Gradio components
             )

demo.launch()