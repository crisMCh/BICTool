BICTool - third-party notices
=============================

BICTool is free software under the GNU General Public License v3 or later (LICENSE.txt).
It contains the following third-party components.


1. Anatomy segmentation network (CNN) and its weights
-----------------------------------------------------

BICTool uses the head-CT neuroanatomy segmentation network and its pretrained weights
(models/weights_mayoiNPH.hdf5, "primary + iNPH" weights) published by the Mayo Clinic
Radiology Informatics Lab: https://github.com/jasonccai/HeadCTSegmentation
(c) 2020, Mayo Clinic Radiology Informatics Lab. BICTool's network definition is based on
that code.

Please cite:
JC Cai, Z Akkus, KA Philbrick, A Boonrod, S Hoodeshenas, AD Weston, P Rouzrokh, GM Conte,
A Zeinoddini, DC Vogelsang, Q Huang, BJ Erickson. "Fully Automated Segmentation of
Neuroanatomy on Head CT Using Deep Learning". Radiol Artif Intell. 2020;2(5):e190183.
https://doi.org/10.1148/ryai.2020190183

The source repository states no license. This package is for internal use; permission to
redistribute the weights and the derived network code has been requested from the authors.


2. Python and Python packages
-----------------------------

The program runs on Python (Python Software Foundation License) and contains these
Python packages. Their license texts are in the folder "licenses".

| Package | Version | License |
|---|---|---|
{{PYTHON_PACKAGES}}

Qt / PySide6 is used under the GNU Lesser General Public License v3 (LGPL-3.0); its
libraries are included as separate, replaceable files (folder _internal\PySide6).
