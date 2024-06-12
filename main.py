from webui.webui import create_webui

demo = create_webui()

if __name__ == '__main__':
    demo.launch(server_port=7880)