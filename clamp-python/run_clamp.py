import numpy as np
import os
from engine import GreedyEngine
from utils import trim_motif, plot_logo_stack, filter_motifs
from input import parse_meme_files
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLAMP on a set of MEME files')
    parser.add_argument('--meme', nargs='+', default=None, help='MEME files to process')
    parser.add_argument('--meme-list', default=None, help='File containing list of MEME files to process')
    parser.add_argument('--nsites-thresh', type=int, default=10, help='Minimum number of sites to consider a motif')
    parser.add_argument('--evalue-thresh', type=float, default=.01, help='Maximum E-value to consider a motif')
    parser.add_argument('--info-score-thresh', type=float, default=5., help='Minimum information score to consider a motif')
    parser.add_argument('--periodicity1-thresh', type=float, default=.6, help='Maximum periodicity (p=1) to consider a motif')
    parser.add_argument('--periodicity2-thresh', type=float, default=.75, help='Maximum periodicity (p=2) to consider a motif')
    parser.add_argument('--periodicity3-thresh', type=float, default=.75, help='Maximum periodicity (p=3) to consider a motif')
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
    
    items, sources, sites = parse_meme_files(meme_files, get_sites=args.get_sites)
    items = filter_motifs(items, nsites_thresh=args.nsites_thresh, evalue_thresh=args.evalue_thresh,
        info_score_thresh=args.info_score_thresh, periodicity1_thresh=args.periodicity1_thresh,
        periodicity2_thresh=args.periodicity2_thresh, periodicity3_thresh=args.periodicity3_thresh)
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
