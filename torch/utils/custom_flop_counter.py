import torch
from torch.utils._pytree import tree_map
from typing import List, Any, Dict, Optional, Union
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from math import prod


aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

def mm_flop(a_shape, b_shape, out=None) -> int:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    return m * n * 2 * k


def addmm_flop(self_shape, a_shape, b_shape, out=None, **kwargs) -> int:
    """
    Count flops for addmm
    """
    return mm_flop(a_shape, b_shape)


def bmm_flop(a_shape, b_shape, out=None, **kwargs) -> int:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop


def baddbmm_flop(self_shape, a_shape, b_shape, out=None, **kwargs) -> int:
    """
    Count flops for the baddbmm operation.
    """
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    return bmm_flop(a_shape, b_shape)


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> int:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for bias are ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *dims = w_shape

    # NB(chilli): I don't think this properly accounts for padding :think:
    # NB(chilli): Should be 2 * c_in - 1 technically for FLOPs.
    flop = batch_size * prod(conv_shape) * c_out * prod(dims) * 2 * c_in
    return flop


def conv_flop(x_shape, w_shape, bias, stride, padding, dilation, transposed, *args, out=None, **kwargs) -> int:
    """
    Count flops for convolution.
    """
    return conv_flop_count(x_shape, w_shape, out, transposed=transposed)


def pointwise_flop(a_shape, *args, **kwargs) -> int:
    """
    Count flops for pointwise operations (addition, subtraction, multiplication, division).
    """
    
    # Calculate the total number of elements in the input tensors by
    num_elements = 2 * prod(a_shape)
    flops = num_elements
    return flops

def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def conv_backward_flop(
        grad_out_shape,
        x_shape,
        w_shape,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
        out) -> int:
    flop_count = 0

    if output_mask[0]:
        grad_input_shape = get_shape(out[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(out[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, transposed)

    return flop_count


def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.
    NB: We can assume that value_shape == key_shape
    """
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and d_q == _d2 and s_k == _s3 and d_q == _d2
    total_flops = 0
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops


def sdpa_flop(query_shape, key_shape, value_shape, *args, out=None, **kwargs) -> int:
    """
    Count flops for self-attention.
    """
    # NB: We aren't accounting for causal attention here
    return sdpa_flop_count(query_shape, key_shape, value_shape)


def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    total_flops = 0
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
    assert d_v == _d4 and s_k == _s3 and s_q == _s4
    total_flops = 0
    # Step 1: We recompute the scores matrix.
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))

    # Step 2: We propagate the gradients through the score @ v operation.
    # gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    # scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))

    # Step 3: We propagate th gradients through the k @ v operation
    # gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    # q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops


def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out=None, **kwargs) -> int:
    """
    Count flops for self-attention backward.
    """
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)

flop_mapping = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop,
}

pointwise_operations = {
    aten.add, aten.mul, aten.div, aten.sub, aten.pow
}

for op in pointwise_operations:
    flop_mapping[op] = pointwise_flop
    


def batch_norm_flop(input_shape, *args, **kwargs) -> int:
    """
    Estimate the number of FLOPs required for a batch normalization operation.

    Args:
        input_shape: A tuple representing the shape of the input tensor. The first dimension is batch size, 
                     the second is number of channels, and the rest are spatial dimensions.
        **kwargs: Additional arguments (ignored).

    Returns:
        An integer representing the estimated number of FLOPs required for a batch normalization operation.
    """
    # print(input_shape)
    if (len(input_shape) < 3):
        return 6 * prod(input_shape)
    batch_size, num_channels = input_shape[:2]
    spatial_dimensions = input_shape[2:]
    
    num_elements = batch_size * num_channels * prod(spatial_dimensions)
    
    mean_flops = num_elements * 2
    variance_flops = num_elements * 4
    
    total_flops = mean_flops + variance_flops
    
    return total_flops


    

flop_mapping.update({
    aten.var_mean: batch_norm_flop,
})

from .._inductor.scheduler import SchedulerNode
from .._inductor.ir import ComputedBuffer, MultiOutputLayout

def get_input_size(inp1):
    if isinstance(inp1.layout, MultiOutputLayout):
        sizes = []
        for inp in inp1.inputs:
            sizes += get_input_size(inp)
        return sizes
    else:
        return [inp1.layout.size]
    

def get_total_flop(nodes):

    
    flop = 0
    for node in nodes:
        node = node.node 
        
        inputs = []
        kwargs = {}
        
        if node.is_extern():
            inputs = []
            for inp in node.inputs:
                inputs += get_input_size(inp)
        
            kwargs.update(node.kwargs)
            try:
                kwargs.update({'out': node.layout.size})
            except:
                pass
        elif isinstance(node, ComputedBuffer):
            if torch._inductor.config.hilea_debug:
                print(node.origins)
            inputs = [node.layout.size]
            try:
                kwargs.update({'out': node.layout.size})
            except:
                pass
            
        for origin in node.origins:
            if torch._inductor.config.hilea_debug:
                print("origin", type(origin), origin)
            # import pprint
            # pprint.pprint(origin.__dict__)
            # print(origin.args)
            node_flop = 0 
            func_name = origin.target
            if isinstance(func_name, torch._ops.OpOverload):
                packet = func_name.overloadpacket
                # print(packet)
                if packet in flop_mapping:
                    #node_flop += flop_mapping[packet](origin.args)
                    try:
                        node_flop = flop_mapping[packet](*inputs, **kwargs)
                    except Exception as e:
                        if torch._inductor.config.hilea_debug:
                            print(e)
                            print(packet)
                            print(inputs)
                            print(kwargs)
                            
                        
                        
            flop += node_flop
    # print(flop)
    return float(flop)
            
            
