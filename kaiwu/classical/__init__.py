# -*- coding: utf-8 -*-
"""
模块: classical

功能: 提供一系列经典求解器

"""
# 作者: 王勇 @QBoson

from kaiwu.classical._simulated_annealing import simulated_annealing
from kaiwu.classical._tabu_search import tabu_search

__all__ = ["simulated_annealing", "tabu_search"]
