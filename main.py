import gradio as gr
from paraphrase.exec import Pegasus
 

if __name__ == '__main__':
    P = Pegasus(num_return_sequences=2, num_beams=2)
    demo = gr.Interface(
        description='''num_beams 参数代表束宽。数值过低可能会导致生成的结果与原文过于相似，而数值过高则可能导致生成的结果偏离原文，出现不可靠的输出。

        num_return_sequences 参数表示生成的输出句子的数量。
        ''',
        fn=P.exec, 
        inputs=[
            gr.Textbox(label="输入文本", placeholder="请输入要处理的文本"), 
            gr.Slider(value=5, minimum=2, maximum=20, step=1, label="num_beams"),
            gr.Slider(value=2, minimum=2, maximum=20, step=1, label="num_return_sequences")], 
        outputs=[
            gr.Textbox(label="全文输出", placeholder="在这里查看完整的输出结果"), 
            gr.Textbox(label="单句选择", placeholder="在这里查看单句输出")
            ],
        
        submit_btn="提交",
        clear_btn="清空",
        allow_flagging = "never"
        )
    demo.launch()