import re
import numpy as np
import os
from engine import GreedyEngine, GreedyItem
from utils import trim_motif, plot_logo_stack
import argparse

def parse_meme_files(meme_files, nsites_thresh=10, evalue_thresh=.1,
    nsites_pattern = re.compile('letter-probability matrix: alength= \d+ w= \d+ nsites= (\d+) E= ([\d.+e-]+)'),
    weights_pattern = re.compile('\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)'),
    width_pattern = re.compile('MOTIF.+width =\s+(\d+)'), get_sites=False):

    items = []
    sources = []
    sites = {}
    n = 0
    for fn in meme_files:
        file_n = 1
        with open(fn, 'r') as f:
            for line in f:
                nsites_match = nsites_pattern.match(line)
                width_match = width_pattern.match(line)
                if width_match:
                    width = int(width_match.groups()[0])
                if line.strip().endswith('sites sorted by position p-value') and get_sites:
                    f.readline()
                    f.readline()
                    f.readline()
                    line = f.readline()
                    sites[n] = set()
                    while line.startswith('chr'):
                        region = line.strip().split('(')[0]
                        offset = int(line[30:37])
                        chrom, coords = region.split(':')
                        start = int(coords.split('-')[0]) + offset - 1
                        stop = start + width
                        sites[n].add((chrom, start, stop, line[29]))
                        line = f.readline()

                if nsites_match:
                    nsites = int(nsites_match.groups()[0])
                    evalue = float(nsites_match.groups()[1])
                    rows = []
                    line = f.readline()
                    weights_match = weights_pattern.match(line)
                    while weights_match:
                        rows.append([float(w) * nsites for w in weights_match.groups()])
                        line = f.readline()
                        weights_match = weights_pattern.match(line)
                    if nsites >= nsites_thresh and evalue <= evalue_thresh:
                        source = ('{}-motif{}'.format(fn, file_n), nsites, evalue)
                        items.append(GreedyItem(n, np.array(rows), source, sites[n] if get_sites else set()))
                        sources.append(source)
                        file_n += 1
                        n += 1
                    elif get_sites:
                        sites.pop(n)
    
    return items, sources, sites


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLAMP on a set of MEME files')
    parser.add_argument('--meme', nargs='+', default=None, help='MEME files to process')
    parser.add_argument('--meme-list', default=None, help='File containing list of MEME files to process')
    parser.add_argument('--nsites-thresh', type=int, default=10, help='Minimum number of sites to consider a motif')
    parser.add_argument('--evalue-thresh', type=float, default=.1, help='Maximum E-value to consider a motif')
    parser.add_argument('--pc', nargs='+', type=float, default=[2., 2., 2., 2.], help='Alpha parameters for Dirichlet prior')
    parser.add_argument('--min-base-overlap', type=int, default=4, help='Minimum number of bases to overlap for merging')
    parser.add_argument('--min-information-overlap', type=float, default=0., help='Minimum information overlap for merging')
    parser.add_argument('--max-information-overhang', type=float, default=12., help='Maximum information overhang for merging')
    parser.add_argument('--concentration', type=float, default=.5, help='Concentration parameter for merging')
    parser.add_argument('--n-processes', type=int, default=None, help='Number of processes to use for parallelization')
    parser.add_argument('--info-thresh', type=float, default=.5, help='Information threshold for trimming motifs')
    parser.add_argument('--get-sites', action='store_true', help='Whether to extract binding sites from MEME files')
    parser.add_argument('--output-dest', '-o', default='clamp_out', help='Folder to save results, will be created if it does not exist')
    args = parser.parse_args()

    if args.meme:
        if args.meme_list:
            print('Warning: both --meme and --meme-list specified, ignoring --meme-list')
        meme_files = args.meme
    elif args.meme_list:
        with open(args.meme_list, 'r') as f:
            meme_files = [line.strip() for line in f.read().strip().split('\n')]
    else:
        parser.error('Either --meme or --meme-list must be specified')

    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest)
    
    items, sources, sites = parse_meme_files(meme_files, args.nsites_thresh, args.evalue_thresh, get_sites=args.get_sites)
    engine = GreedyEngine(items, pc=args.pc, min_base_overlap=args.min_base_overlap,
        min_information_overlap=args.min_information_overlap,
        max_information_overhang=args.max_information_overhang, concentration=args.concentration)
    engine.cluster_motifs(n_processes=args.n_processes)

    maximal_clusters = engine.clusters_trace[np.argmax(engine.llr_trace)]
    for c in maximal_clusters:
        # Create a directory for each cluster
        if not os.path.exists('{}/cluster{}'.format(args.output_dest, c)):
            os.mkdir('{}/cluster{}'.format(args.output_dest, c))

        cluster = engine.clusters[c]

        # Write the aligned PFMs to a TRANSFAC file
        with open('{0}/cluster{1}/cluster{1}_aligned-motifs.transfac'.format(args.output_dest, c), 'w') as f:
            for i in range(len(cluster.items)):
                motif_id = sources[cluster.items[i].idx][0]
                f.write('AC\t{}\n'.format(motif_id))
                f.write('XX\n')
                f.write('ID\t{}\n'.format(motif_id))
                f.write('PO\tA\tC\tG\tT\n')
                pfm = cluster.aligned_pfms[i, :, :]
                for j in range(pfm.shape[0]):
                    f.write('{:02d}\t{:06f}\t{:06f}\t{:06f}\t{:06f}\n'.format(j + 1, *pfm[j, :]))
                f.write('XX\n//\n')

        # Write the consensus PFM to a TRANSFAC file
        with open('{0}/cluster{1}/cluster{1}_consensus-pfm.transfac'.format(args.output_dest, c), 'w') as f:
            f.write('AC\t{}\n'.format('cluster{}'.format(c)))
            f.write('XX\n')
            f.write('ID\t{}\n'.format('cluster{}'.format(c)))
            f.write('PO\tA\tC\tG\tT\n')
            trimmed_pfm = trim_motif(cluster.aligned_pfms, args.info_thresh)
            for j in range(trimmed_pfm.shape[0]):
                f.write('{:02d}\t{:06f}\t{:06f}\t{:06f}\t{:06f}\n'.format(j + 1, *trimmed_pfm[j, :]))
            f.write('XX\n//\n')

        # Plot the aligned PFMs as an SVG
        svg = plot_logo_stack(cluster.aligned_pfms)
        with open('{0}/cluster{1}/cluster{1}.svg'.format(args.output_dest, c), 'w') as f:
            svg.writexml(f, addindent='\t', newl='\n')
