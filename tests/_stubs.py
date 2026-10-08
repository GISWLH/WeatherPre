"""Stand-in networks with the official interfaces (no weights). They make the adapter plumbing testable; their outputs are
not forecasts."""
import os
import numpy as np

ORCA_EXAMPLE = os.environ.get("ORCA_EXAMPLE", "/home/user/data/orca_example")
FUXI_SAMPLE = os.environ.get("FUXI_SAMPLE", "/home/user/data/fuxi_s2s_sample")


def orca_persistence_forecaster(example_dir):
    """ORCADLForecaster on the official grid whose 'network' returns normalised zeros and whose statistics are mean = the input
    state, std = 1: every lead de-normalises to the initial state (persistence). Exercises the real normalisation / masks."""
    import torch
    from weatherai.inference import orca_dl as O
    from weatherai.models.orca_dl import ORCADLConfig
    inp = O.ORCADLInputs.from_official_example(example_dir, "1980-01")

    class Zero(torch.nn.Module):
        def forward(self, o, a, predict_time_steps=1):
            return torch.zeros(o.shape[0], predict_time_steps, *o.shape[1:])

    st = {"mean": {}, "std": {}}
    for v in O.OCEAN_VARS + O.ATMO_VARS:
        a = np.asarray(inp.fields[v], np.float32)
        a = a[None] if a.ndim == 2 else a
        st["mean"][v] = np.broadcast_to(a, (12,) + a.shape)
        st["std"][v] = np.ones((12,) + a.shape, np.float32)
    return O.ORCADLForecaster.from_modules({1: Zero(), 2: Zero()}, st, ORCADLConfig.official(), O.ORCA_DEPTHS), inp


def fuxi_stub_onnx(root):
    """ONNX graph with the official FuXi-S2S interface: pred_k = last + 0.01*k + mean(eps1) + mean(eps2) (finite over land)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    d = os.path.join(root, "model-1.0"); os.makedirs(d, exist_ok=True)
    c = lambda n, v: helper.make_node("Constant", [], [n], value=numpy_helper.from_array(np.asarray(v)))
    nodes = [helper.make_node("RandomNormalLike", ["like1"], ["/rn1"], dtype=TensorProto.DOUBLE),
             helper.make_node("RandomNormalLike", ["like2"], ["/rn2"], dtype=TensorProto.DOUBLE),
             helper.make_node("ReduceMean", ["/rn1"], ["m1"], keepdims=0), helper.make_node("ReduceMean", ["/rn2"], ["m2"], keepdims=0),
             helper.make_node("Add", ["m1", "m2"], ["m12"]), helper.make_node("Cast", ["m12"], ["noise"], to=TensorProto.FLOAT),
             c("s1", np.array([1], np.int64)), c("s2", np.array([2], np.int64)), c("ax", np.array([1], np.int64)),
             helper.make_node("Slice", ["input", "s1", "s2", "ax"], ["last"]), helper.make_node("IsNaN", ["last"], ["isnan"]),
             c("zero", np.array(0.0, np.float32)), helper.make_node("Where", ["isnan", "zero", "last"], ["lastfix"]),
             c("c01", np.array(0.01, np.float32)), helper.make_node("Mul", ["step", "c01"], ["sterm"]),
             helper.make_node("Add", ["lastfix", "sterm"], ["p1"]), helper.make_node("Add", ["p1", "noise"], ["pred"]),
             helper.make_node("Concat", ["last", "pred"], ["output"], axis=1)]
    g = helper.make_graph(nodes, "stub", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 76, 121, 240]),
                                          helper.make_tensor_value_info("step", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 76, 121, 240])],
                          initializer=[numpy_helper.from_array(np.zeros((128, 12), np.float32), "like1"),
                                       numpy_helper.from_array(np.zeros((128, 7200), np.float32), "like2")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, os.path.join(d, "fuxi_s2s.onnx"))
    return root
