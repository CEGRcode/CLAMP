# CLAMP: identification of consensus sequence motifs from genomic-scale data

### Justin S Cha<sup>1</sup>, B Franklin Pugh<sup>1</sup>, William KM Lai<sup>1,2*</sup>

<sup>1</sup>Department of Molecular Biology and Genetics, Cornell University, USA
<br>
<sup>2</sup>Department of Computational Biology, Cornell University, USA

### Correspondence: wkl29@cornell.edu

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
Example command on simulated motifs:
```
python clamp-python/run_clamp.py --meme simulated_motifs/SIM-1.meme --output-dest simulated_motifs/SIM-1_results
```
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

### Flags
 - `--meme`: Input MEME files
 - `--meme-list`: TXT file with paths to input MEME files separated by newlines
   - One of `--meme` or `--meme-list` is required
 - `--nsites-thresh`: Motifs with nsites less than the passed in value will be filtered out
 - `--evalue-thresh`: Motifs with E-value greater than the passed in value will be filtered out
 - `--info-score-thresh`: Motifs with information score less than the passed in value will be filtered out
 - `--periodicity1-thresh`: Motifs with periodicity score for period 1 greater than the passed in value will be filtered out
 - `--periodicity2-thresh`: Motifs with periodicity score for period 2 greater than the passed in value will be filtered out
 - `--periodicity3-thresh`: Motifs with periodicity score for period 3 greater than the passed in value will be filtered out
 - `--pc`: Pseudocounts for prior Dirichlet background model
 - `--min-base-overlap`: Minimum number of overlapping bases allowed for merging clusters
 - `--min-information-overlap`: Minimum bit overlap dot product allowed for merging clusters
 - `--max-information-overhang`: Maximum sum of absolute bit difference allowed for merging clusters
 - `--concentration`: Concentration parameter (clustering score = BLLR * cluster_size ^ concentration)
 - `--n-workers`: Number of worker threads (default is number of CPUs)
 - `--trim-thresh`: Bases on the periphery of the consensus motif with information below the passed in value will be trimmed off
 - `--get-sites`: Include to output bed files of union of binding sites for each cluster (input MEME files must include site information)
 - `--output-dest`: Output directory

### Output
CLAMP will create folders inside the folder specified by `--output-dest` with names formatted as `clusterX`. Each folder will contain three files:

 - Aligned PFMs in transfac format named `clusterX_aligned-motifs.transfac`
 - Consensus PFM in transfac format named `clusterX_consensus-motif.transfac`
 - Consensus PFM in meme format named `clusterX_consensus-motif.meme`
 - Aligned stack of logos as an SVG named `clusterX_aligned-motifs.svg`
 - Consensus logo as an SVG named `clusterX_consensus-motif.svg`
 - Optional bed file of binding sites named `clusterX_binding-sites.bed`