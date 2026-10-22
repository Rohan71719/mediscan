from torchinfo import summary
from src.models import ArogyaNet
from src.utils import OUT

if __name__ == "__main__":
    s = summary(ArogyaNet(), input_size=(1, 1, 224, 224), col_names=("output_size", "num_params"),
                depth=3, verbose=0)
    print(s); (OUT / "arch_table.txt").write_text(str(s))
    for name, kw in {"full": {}, "no_res": dict(residual=False), "no_se": dict(se=False),
                     "two-branch": dict(meta=True)}.items():
        print(f"{name:12s} params = {sum(p.numel() for p in ArogyaNet(**kw).parameters()):,}")
