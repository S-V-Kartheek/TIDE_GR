"""Method registry: every method (TIDE-GR and all baselines) shares one interface."""


def build_method(cfg, data, device, seed):
    kind = cfg.method.get("kind", "graphrec")
    if kind == "graphrec":
        from .graphrec import GraphRec
        return GraphRec(cfg, data, device, seed)
    if kind == "ensemble":
        from .ensemble import Ensemble
        return Ensemble(cfg, data, device, seed)
    if kind in ("sasrec", "sqn"):
        from .sasrec import SeqRec
        return SeqRec(cfg, data, device, seed)
    raise ValueError(f"unknown method kind {kind}")
