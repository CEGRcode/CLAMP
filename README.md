# CLAMP: identification of consensus sequence motifs from genomic-scale data

### Justin S Cha<sup>1</sup>, William KM Lai<sup>2</sup>, B Franklin Pugh<sup>2</sup>

<sup>1</sup>Department of Computational Biology, Cornell University, USA
<br>
<sup>2</sup>Department of Molecular Biology and Genetics, Cornell University, USA

### Correspondence: fp265@cornell.edu

## Abstract
Motif databases and motif calls on large-scale genomic protein binding experiments contain many highly similar, redundant motifs. Consolidating sets of redundant motifs to their consensuses minimizes the space of motifs and relates the corresponding transcription factors (TFs). Previous efforts to consolidate redundant motifs have various issues that result in suboptimal consensus motifs. CLAMP improves upon previous methods by coupling clustering with motif alignment in a single combined algorithm. We demonstrate this improvement by benchmarking CLAMP against other methods on simulated motifs. We run CLAMP on a set of motifs derived from large-scale S. cerevisiae ChIP-exo datasets to demonstrate its applicability in constructing sets of reference points and aggregating evidence for novel motifs.

## Software details
### Dependencies
 - python >= 3.6
 - numpy
 - scipy
 - numba

### Getting started
To download CLAMP:
```
git clone https://github.com/CEGRcode/CLAMP.git
```
Support on pypi/conda coming soon

### Running CLAMP
CLAMP takes in a list of motifs using the MEME file format. This list can be inputted using `--meme`:
```
python clamp-python/run_clamp.py --meme motif1.meme motif2.meme ... {other_args}
```
They can also be inputted as a newline-delimited text file using `--meme-list`:
```
python clamp-python/run_clamp.py --meme-list meme_files.txt {other_args}
```
The default output location is `./clamp_out`. This can be changed with `--output-dest` or `-o`:
```
python clamp-python/run_clamp.py --output-dest new_clamp_out {other_args}
```

### Output
CLAMP will create folders inside the folder specified by `--output-dest` with names formatted as `clusterX`. Each folder will contain three files:

 - Aligned PFMs in transfac format named `clusterX_aligned-motifs.transfac`
 - Consensus PFM in transfac format named `clusterX_consensus-pfm.transfac`
 - Aligned stack of logos as an SVG named `clusterX.svg`