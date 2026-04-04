# 这个文件只是一个极薄的启动入口。
# 真正的命令行逻辑都在 `src/langgraph_lightrag_demo/cli.py` 里。
# 保留这个文件的好处是：你可以继续用 `python main.py ...` 启动项目。
from langgraph_lightrag_demo.cli import main


if __name__ == "__main__":
    main()
