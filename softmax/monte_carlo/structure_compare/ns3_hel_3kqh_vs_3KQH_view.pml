load 3KQH.pdb, exp
load ns3_hel_3kqh_vs_3KQH_superposed.pdb, pred
remove solvent
remove exp and not chain A
hide everything
show cartoon
color grey70, exp
spectrum b, red_white_blue, pred, minimum=30, maximum=90
# already superposed by compare_structures.py. To redo the fit inside PyMOL:
#   align pred and name CA, exp and name CA, cycles=0    (no outlier rejection)
#   align pred and name CA, exp and name CA              (PyMOL default, drops outliers)
orient
bg_color white
