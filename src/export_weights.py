"""
Сохраняет дообученные энкодеры в fp16 в ../weights/ - для выкладки рядом с кодом.

Эмбеддинги всё равно считаются в fp16 (Encoder(...).half()), так что результат
с этими весами тот же, а весят они вдвое меньше.
"""
from transformers import AutoModel, AutoTokenizer

from common import ROOT, WORK

MODELS = ["e5s_hn", "e5b", "e5b_hn"]


def main():
    out = ROOT / "weights"
    for name in MODELS:
        src = WORK / f"dense_{name}"
        model = AutoModel.from_pretrained(src).half()
        model.save_pretrained(out / f"dense_{name}")
        AutoTokenizer.from_pretrained(src).save_pretrained(out / f"dense_{name}")
        print("saved", out / f"dense_{name}")


if __name__ == "__main__":
    main()
