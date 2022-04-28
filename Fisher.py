# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:13:58 2022

@author: Gordon Huang
"""

from scipy.stats import fisher_exact
'''
<

'''
table = [[79, 241], [10, 57]]#1-years, AHF score 0.6
oddsr, p = fisher_exact(table, alternative='less')
print(oddsr, p)

table = [[55, 159], [34, 139]]#5-years, AHF score 0.6
oddsr, p = fisher_exact(table, alternative='less')
print(oddsr, p)

table = [[28, 65], [61, 233]]#10-years, AHF score 0.6
oddsr, p = fisher_exact(table, alternative='less')
print(oddsr, p)

table = [[155, 165], [24, 43]]#1-years, AHF score 0.8
oddsr, p = fisher_exact(table, alternative='greater')
print(oddsr, p)

table = [[104, 110], [75, 98]]#5-years, AHF score 0.8
oddsr, p = fisher_exact(table, alternative='less')
print(oddsr, p)

table = [[166, 42], [128, 51]]#10-years, AHF score 0.8
oddsr, p = fisher_exact(table, alternative='less')
print(oddsr, p)
