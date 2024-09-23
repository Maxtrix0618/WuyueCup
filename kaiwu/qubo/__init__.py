# -*- coding: utf-8 -*-
"""
模块: qubo

功能: 提供一系列针对QUBO的前处理工具

"""

from kaiwu.qubo._interface import details, qubo_matrix_to_qubo_model
from kaiwu.qubo._interface import binary, spin, make, sum, constraint, matrix, cim_ising_model, ising_quadratization
from kaiwu.qubo._ndarray import ndarray
from kaiwu.qubo._qubo_dict import qubo_dict
from kaiwu.qubo._get_val import get_sol_dict, get_val, get_array_val, get_dict_val
from kaiwu.qubo._dynamic_range import check_qubo_bit_width, adjust_matrix_precision

__all__ = ["binary", "spin", "make", "sum", "constraint", "matrix", "cim_ising_model", "ising_quadratization",
           "details", "qubo_matrix_to_qubo_model"]
__all__ += ["ndarray"]
__all__ += ["qubo_dict"]
__all__ += ["get_sol_dict", "get_val", "get_array_val", "get_dict_val"]
__all__ += ["check_qubo_bit_width", "adjust_matrix_precision"]
