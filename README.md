# CLAMP: identification of consensus sequence motifs from genomic-scale data

### Justin S Cha<sup>1</sup>, William KM Lai<sup>2</sup>, B Franklin Pugh<sup>2</sup>

<sup>1</sup>Department of Computational Biology, Cornell University, USA
<br>
<sup>2</sup>Department of Molecular Biology and Genetics, Cornell University, USA

### Correspondence: fp265@cornell.edu

## Abstract
Genome-wide protein-DNA genome-wide mapping projects have generated thousands of consensus motifs representing enriched DNA-sequence motifs. Many of these motifs are redundant or highly similar to each other due to transcription factors (TFs) binding as complexes to the same motif. Consolidating these motifs into representative consensus profiles is critical to reduce redundancy and better characterize TF-DNA relationships. We present CLustered Alignment of Motif Profiles (CLAMP), a novel algorithm that condenses redundant motifs into consensus motifs by coupling motif clustering and alignment in a single unified framework. We validate the utility of CLAMP using a set of motifs derived from a large-scale S. cerevisiae ChIP-exo mapping project. We identify previously discovered protein complexes binding at shared motifs as well as evidence of novel motifs. We also find that CLAMP-derived consensus motifs demonstrate positional and combinatorial arrangement at specific genomic features constituting a DNA-sequence ‘grammar’ throughout the genome. Together, these results establish CLAMP as a robust, efficient method for motif consolidation that enhances both computational analyses and biological interpretation of large-scale TF binding data.

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