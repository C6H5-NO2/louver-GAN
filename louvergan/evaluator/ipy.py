try:
    from IPython.display import Markdown, display
except ModuleNotFoundError:
    Markdown = str
    display = print
