# reference/ - local data, not in git

This folder holds files BICTool needs or was checked against that must **not** be committed: the CNN weights
(third-party, no license), volumes derived from real patient scans, and the original Octave code. Git ignores
everything here except this README. Never `git add -f` anything from this folder.

## CNN weights for the anatomy segmentation (required)

BICTool's anatomy segmentation uses the network of the Mayo Clinic Radiology Informatics Lab:
Cai JC, Akkus Z, Philbrick KA, et al. *Fully Automated Segmentation of Head CT Neuroanatomy Using Deep
Learning.* Radiol Artif Intell. 2020;2(5):e190183. https://doi.org/10.1148/ryai.2020190183

The weights are not in this repository. Download them yourself:

1. Open the Mayo repository: https://github.com/jasonccai/HeadCTSegmentation (section *Installation
   Instructions*, step 5).
2. Download the weights **"for the primary dataset and the iNPH dataset (... recommended for routine use)"**:
   https://drive.google.com/file/d/1fSe7oDaE8NYh5GLUN6h-Yo-ZlrJHXeuZ/view?usp=sharing
   The file is `weights_mayoiNPH.hdf5` (about 373 MB). Do **not** use the other file, `weights_mayo40.hdf5`
   (training dataset only).
3. Save it as `reference/segmentation/weights_mayoiNPH.hdf5` in your main checkout (or anywhere outside the
   repository).
4. Tell BICTool where it is, in `~/.bictool/config.json` (Windows: `C:\Users\<name>\.bictool\config.json`;
   create the folder and file if they do not exist):

   ```json
   {
     "segmentation": {
       "anatomy_weights_path": "E:/path/to/BICTool/reference/segmentation/weights_mayoiNPH.hdf5"
     }
   }
   ```

   Use forward slashes or doubled backslashes. Alternatively set the environment variable
   `BICTOOL_HEADCT_WEIGHTS` to the file path (it wins over the config file).
5. Start BICTool and open a study: the *Anatomical* box shows the segmentation progress. Without weights it
   says *"Anatomy segmentation weights not found"*.

The weights source states no license. They may be used internally; BfS has asked Mayo for permission to
redistribute them (see `packaging/THIRD_PARTY_NOTICES.md`). The tests do not need the weights.

## Other contents (only on machines that have them)

| Folder | Content |
|---|---|
| `segmentation/` | the weights; reference volumes and CNN predictions of two real scans, used once to confirm the label order (`domain/segmentation/head_ct_unet.py`) - patient-derived, never commit or share |
| `octave/` | the original Octave implementation the domain code was ported from (Passand 2022) |
