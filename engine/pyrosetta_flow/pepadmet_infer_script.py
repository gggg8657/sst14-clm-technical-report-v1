"""pepADMET inference script — pepadmet conda env에서 실행."""
import sys, os, json, types

REPO = os.environ.get("PEPADMET_REPO", "[LOCAL_PATH]")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
os.environ["DGLBACKEND"] = "pytorch"

mock_wv = types.ModuleType("weight_visualization")
mock_wv.weight_visualize = lambda *a, **k: None
sys.modules["utils.weight_visualization"] = mock_wv
sys.modules["weight_visualization"] = mock_wv

import torch, dgl, numpy as np
from rdkit import Chem
from build_dataset import atom_features, etype_features
from MY_GNN import MGA
from calculate_descriptors import calculate_descriptors

DESCRIPTOR_DIM = 2133


def build_descriptor_tensor(smiles: str, seq: str) -> torch.Tensor:
    """calculate_descriptors()로 실제 2133차원 descriptor tensor를 생성한다.

    차원이 맞지 않으면 zero-padding 또는 truncation으로 보정한다.
    계산 실패 시 zero tensor를 반환한다.
    """
    try:
        fp = calculate_descriptors((smiles, seq))
        if "Error" in fp:
            return torch.zeros(1, DESCRIPTOR_DIM, dtype=torch.float32)

        feature_values = [
            float(v)
            for k, v in sorted(fp.items())
            if isinstance(v, (int, float)) and k not in ("SMILES", "SEQUENCE")
        ]

        n = len(feature_values)
        if n < DESCRIPTOR_DIM:
            feature_values += [0.0] * (DESCRIPTOR_DIM - n)
        elif n > DESCRIPTOR_DIM:
            feature_values = feature_values[:DESCRIPTOR_DIM]

        return torch.tensor([feature_values], dtype=torch.float32)
    except Exception:
        return torch.zeros(1, DESCRIPTOR_DIM, dtype=torch.float32)


def graph_from_mol(mol):
    """RDKit Mol → DGL 그래프 (pepADMET 학습 시와 동일한 atom/edge 특성)."""
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        a_feats = np.array([atom_features(a) for a in mol.GetAtoms()])
        src, dst, etypes = [], [], []
        for b in mol.GetBonds():
            src += [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            dst += [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
            et = etype_features(b)
            etypes += [et, et]
        g = dgl.DGLGraph()
        g.add_nodes(mol.GetNumAtoms())
        g.add_edges(src, dst)
        g.ndata["atom"] = torch.tensor(a_feats, dtype=torch.float32)
        g.edata["etype"] = torch.tensor(etypes, dtype=torch.long)
        return g
    except Exception:
        return None


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return graph_from_mol(mol)


def graph_from_sequence_linear(seq: str):
    """SMILES 파싱 실패 시 선형 펩타이드(MolFromSequence)로 그래프 생성.

    사이클릭/이황화 SMILES가 RDKit 버전에 따라 깨지는 경우가 있어 폴백으로 사용한다.
    """
    if not seq or not isinstance(seq, str):
        return None, None
    s = seq.upper().strip()
    try:
        mol = Chem.MolFromSequence(s)
    except Exception:
        mol = None
    if mol is None:
        return None, None
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        smi = ""
    g = graph_from_mol(mol)
    if g is None:
        return None, None
    return g, smi or ""


model_path = os.path.join(REPO, "model", "toxicity_early_stop.pth")
model = MGA(
    in_feats=40, descriptor=2133, descriptor_dim=2133,
    rgcn_hidden_feats=[64, 64], n_tasks=4, rgcn_drop_out=0.2,
    fpn_out=2133, fp_2_dim=512, hidden_size=256,
    select_task_list=["toxicity_nontoxicity", "toxicity_type_class",
                      "neurotoxicity_type_class", "HC50"],
    device="cpu", classifier_hidden_feats=320, dropout=0.2, loop=True,
)
st = torch.load(model_path, map_location="cpu")["model_state_dict"]
model.load_state_dict(st)
model.eval()

TYPE_NAMES = ["cytolysis", "GPCR_toxin", "neurotoxin", "cytotoxicity", "hemostasis", "hemolysis"]
NEURO_NAMES = ["AChR_inhibitor", "Ca_inhibitor", "K_inhibitor", "Na_inhibitor"]

input_data = json.loads(sys.argv[1])
results = []

for item in input_data:
    seq = item["sequence"]
    smiles = item.get("smiles", "") or ""
    g = smiles_to_graph(smiles) if smiles else None
    smiles_used = smiles
    graph_note = None

    if g is None:
        g, smi_fb = graph_from_sequence_linear(seq)
        if g is not None:
            smiles_used = smi_fb if smi_fb else smiles
            graph_note = "linear_sequence_fallback"

    if g is None:
        results.append({
            "sequence": seq,
            "error": "graph build failed (SMILES parse failed and linear sequence fallback failed)",
            "available": False,
        })
        continue

    desc = build_descriptor_tensor(smiles_used, seq)
    bg = dgl.batch([g])

    with torch.no_grad():
        out = model(bg, bg.ndata["atom"], bg.edata["etype"], desc)

    bp = 1 / (1 + np.exp(-out["task_0"].item()))
    tp = torch.softmax(out["task_1"], dim=-1)[0].tolist()
    np_ = torch.softmax(out["task_2"], dim=-1)[0].tolist()
    hc = out["task_3"].item()

    out_row = {
        "sequence": seq, "available": True,
        "binary_toxicity": round(bp, 4),
        "is_toxic": bool(bp > 0.5),
        "toxicity_type": TYPE_NAMES[int(np.argmax(tp))] if int(np.argmax(tp)) < len(TYPE_NAMES) else f"class_{int(np.argmax(tp))}",
        "toxicity_type_confidence": round(max(tp), 4),
        "neurotoxicity_type": NEURO_NAMES[int(np.argmax(np_))] if int(np.argmax(np_)) < len(NEURO_NAMES) else f"class_{int(np.argmax(np_))}",
        "neurotoxicity_confidence": round(max(np_), 4),
        "hc50": round(hc, 4),
    }
    if graph_note:
        out_row["graph_note"] = graph_note
    results.append(out_row)

print(json.dumps(results))
