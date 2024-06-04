import gradio as gr
from paraphrase.exec import Pegasus
 

if __name__ == '__main__':
    P = Pegasus(num_return_sequences=2, num_beams=2)
    demo = gr.Interface(
        description='''
        ```shell
        🎉使用说明🎉
        1. num_beams 该数值过低可能会导致生成的结果与原文过于相似，而数值过高则可能导致生成的结果偏离原文，出现不可靠的输出。
        2. num_return_sequences 参数控制[单句选择]中句子的输出数量。
        3. 两个参数的取值范围均为2~20。提升这两个数值会进行更多推演计算, 耗时相对拉长。
        
        ❗️注意❗️
        1. 程序会对待重写内容进行分句, 分句依据是👉️以英文大写字母, 且以.?!结尾的句子👈️。以下为正确和错误的例子。
            ✅ "Hello world."
            ❌ "hello world."
        2. 原文如有存在括号的内容, 需要后期手工补全。
        3. 程序会逐一对句子进行重写, 并选取最长的句子作为最终输出。其他候选句子均会在[单句选择]输出结果中展示。

        ```
        ''',
        fn=P.exec, 
        inputs=[
            gr.Textbox(label="待重写内容", placeholder="输入或粘贴您要重写的内容, 并按提交按钮"), 
            gr.Slider(value=5, minimum=2, maximum=20, step=1, label="num_beams"),
            gr.Slider(value=4, minimum=2, maximum=20, step=1, label="num_return_sequences")], 
        outputs=[
            gr.Textbox(label="重写结果", placeholder="这里查看最终的输出结果"), 
            gr.Textbox(label="单句结果", placeholder="这里查看单句所有输出结果")
            ],
        
        submit_btn="提交",
        clear_btn="清空",
        allow_flagging = "never"
        )
    demo.launch(server_port=7880)