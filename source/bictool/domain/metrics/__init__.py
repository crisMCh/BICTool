"""Metrics sub-module of the Domain Layer.

Computes image-quality metrics: MTF, NPS, PS, PS/MTF ratio, and other
task-based metrics (FR-03). `mtf`, `nps`, `contrast`, and `ps_mtf_ratio`
are translations of the legacy Octave pipeline (`reference/octave/`);
numeric parity against the original Octave implementation is still being
validated for all of them. The legacy patch-based NPS/SVM-training cluster
(`funcCalcNPS`, `funcPatchAnalysis`, `funcFindLargestSquares`, `funcZigZag`,
`OctaveFeatEstimation`, `funcLoadDicomFile`, `trainSVMWithThesisData`) is not
translated - nothing in the live pipeline (`funcImageQualityOctave.m`) calls
it any more; porting it is planned separately (see the project plan).
"""
