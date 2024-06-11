import gradio as gr

with gr.Blocks() as demo:
    # text_count = gr.Slider(1, 5, step=1, label="Textbox Count")
    num_return_sequences = gr.Slider(value=4, minimum=2, maximum=20, step=1, label="num_return_sequences", info="控制[单句选择]中句子的输出数量。")
    num_beams = gr.Slider(value=10, minimum=2, maximum=20, step=1, label="num_beams", info="该数值过低可能会导致生成的结果与原文过于相似，而数值过高则可能导致生成的结果偏离原文，出现不可靠的输出。")
    # num_return_sequences = gr.Slider(1, 5, step=1, label="Textbox Count")

    
    @gr.render(inputs=[num_beams], triggers=[num_beams.change])
    def render_count(count):
        boxes = []
        for i in range(0,count):
            with gr.Tab(f"Tab {i}", visible=True):
                box = gr.Textbox(key=i, label=f"Box {i}")
                boxes.append(box)

        def merge(*args):
            return [1,2,3]
        
        merge_btn.click(merge, boxes, boxes)

        def clear():
            return [""] * count
                
        clear_btn.click(clear, None, boxes)

        def countup():
            return [i for i in range(count)]
        
        count_btn.click(countup, None, boxes)

    with gr.Row():
        merge_btn = gr.Button("Merge")
        clear_btn = gr.Button("Clear")
        count_btn = gr.Button("Count")
        
    output = gr.Textbox()
demo.launch()