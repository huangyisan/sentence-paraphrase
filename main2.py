import gradio as gr
from paraphrase.exec import Pegasus
from loguru import logger
import uuid

# print gradio version
logger.info("gradio version: {}".format(gr.__version__))
def generate_trace_id():
    return str(uuid.uuid4())

logger.add("log/paraphrase.log", format="{time} | {level} | {message} | trace_id={extra}", rotation="1 MB", level="DEBUG")
 
P = Pegasus()
def paraphrase(text, num_beams, num_return_sequences):
    trace_id = generate_trace_id()
    with logger.contextualize(trace_id=trace_id):
        logger.info(f"Origin text: {text}")
        return ["c","c","c"]
        # return P.exec(text, num_beams, num_return_sequences)
def create_webui():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('''
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
            ''')
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="待重写内容",placeholder="输入或粘贴您要重写的内容, 并按提交按钮", lines=10)
                # 滑条
                num_beams = gr.Slider(value=10, minimum=2, maximum=20, step=1, label="num_beams", info="该数值过低可能会导致生成的结果与原文过于相似，而数值过高则可能导致生成的结果偏离原文，出现不可靠的输出。")
                num_return_sequences = gr.Slider(value=4, minimum=2, maximum=20, step=1, label="num_return_sequences", info="控制[单句选择]中句子的输出数量。")
                # num_return_sequences = gr.Slider(1, 5, step=1, label="Textbox Count")

                with gr.Row():
                    # 清空内容按钮
                    clear_button = gr.Button("清空内容")
                    # 提交按钮
                    submit_button = gr.Button("提交内容")

            with gr.Column():
                
                @gr.render(inputs=[num_return_sequences], triggers=[num_return_sequences.change])
                def render_tag_by_num_return_sequences(count):
                    boxes = []
                    for i in range(count):
                        with gr.Tab(label=f"{i+1}句选择"):
                            # gr.Markdown(f"{i+1}句选择")
                            box = gr.Textbox(key=i, label="重写结果", placeholder="这里查看最终的输出结果")
                            boxes.append(box)
                            # 复制文本按钮
                            gr.Button("复制文本", elem_id=f"copy-{i}")
                    # 按钮作用
                    print("执行submit_butto ")
                    submit_button.click(
                        fn=paraphrase,
                            inputs = [input_text, num_beams, num_return_sequences], 
                            outputs=boxes
                        )
                    print("结束执行submit_butto ")
                    clear_button.click(
                        fn=lambda: ("", ""),
                        inputs=[],
                        outputs=[input_text]
                    )
        
    return demo

demo = create_webui()
if __name__ == '__main__':
    
    demo.launch(server_port=7860)