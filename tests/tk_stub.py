"""A stand-in for tkinter, so that the logic of the logger window can be run without a display."""
import sys
import types
from unittest.mock import MagicMock


class _Var:
    default = None

    def __init__(self, value=None, **_):
        self.v = self.default if value is None else value

    def get(self):
        return self.v

    def set(self, v):
        self.v = v

    def trace_add(self, *_):
        pass


class _Toplevel:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def install():
    tk = types.ModuleType("tkinter")
    tk.StringVar = type("StringVar", (_Var,), {"default": ""})
    tk.IntVar = type("IntVar", (_Var,), {"default": 0})
    tk.BooleanVar = type("BooleanVar", (_Var,), {"default": False})
    tk.Tk = lambda *a, **k: MagicMock()
    tk.Canvas = lambda *a, **k: MagicMock()
    tk.Toplevel = _Toplevel
    ttk, dialog, box = MagicMock(), MagicMock(), MagicMock()
    box.askyesno.return_value = True
    tk.ttk, tk.filedialog, tk.messagebox = ttk, dialog, box
    sys.modules.update({"tkinter": tk, "tkinter.ttk": ttk, "tkinter.filedialog": dialog, "tkinter.messagebox": box})
    return box
