import re
import numpy as np
import os
from engine import GreedyEngine, GreedyItem
from utils import trim_motif, plot_logo_stack
import argparse

def parse_meme_files(meme_files, nsites_thresh=10, evalue_thresh=.1,
    nsites_pattern = re.compile('letter-probability matrix: alength= \d+ w= \d+ nsites= (\d+) E= ([\d.+e-]+)'),
    weights_pattern = re.compile('\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)')):

    items = []
    sources = []
    n = 0
    for fn in meme_files:
        with open(fn, 'r') as f:
            for line in f:
                nsites_match = nsites_pattern.match(line)
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
                        items.append(GreedyItem(n, np.array(rows)))
                        sources.append(('{}-motif{}'.format(fn, n), nsites, evalue))
                        n += 1
    
    return items, sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLAMP on a set of MEME files')
    parser.add_argument('--meme', nargs='+', default=None, help='MEME files to process')
    parser.add_argument('--meme-list', default=None, help='File containing list of MEME files to process')
    parser.add_argument('--nsites-thresh', type=int, default=10, help='Minimum number of sites to consider a motif')
    parser.add_argument('--evalue-thresh', type=float, default=.1, help='Maximum E-value to consider a motif')
    parser.add_argument('--pc', nargs=4, type=float, default=[1., 1., 1., 1.], help='Prior counts for PWM')
    parser.add_argument('--min-information-overlap', type=float, default=8., help='Minimum information overlap for merging')
    parser.add_argument('--max-information-overhang', type=float, default=12., help='Maximum information overhang for merging')
    parser.add_argument('--concentration', type=float, default=1., help='Concentration parameter for merging')
    parser.add_argument('--n-processes', type=int, default=None, help='Number of processes to use for parallelization')
    parser.add_argument('--info-thresh', type=float, default=.5, help='Information threshold for trimming motifs')
    parser.add_argument('--output-dest', '-o', default='clamp_out', help='Folder to save results, will be created if it does not exist')
    args = parser.parse_args()

    if args.meme:
        if args.meme_list:
            print('Warning: both --meme and --meme-list specified, ignoring --meme-list')
        meme_files = args.meme
    elif args.meme_list:
        with open(args.meme_list, 'r') as f:
            meme_files = [line.strip() for line in f]
    else:
        parser.error('Either --meme or --meme-list must be specified')

    if not os.path.exists(args.output_dest):
        os.makedirs(args.output_dest)
    if not os.path.exists('{}/svgs'.format(args.output_dest)):
        os.mkdir('{}/svgs'.format(args.output_dest))
    
    items, sources = parse_meme_files(meme_files, args.nsites_thresh, args.evalue_thresh)
    engine = GreedyEngine(items, pc=args.pc, min_information_overlap=args.min_information_overlap,
        max_information_overhang=args.max_information_overhang, concentration=args.concentration)
    engine.cluster_motifs(n_processes=args.n_processes)

    maximal_clusters = engine.clusters_trace[np.argmax(engine.llr_trace)]
    with open('{}/matrices.chen'.format(args.output_dest), 'w') as chen:
        with open('{}/motif_clusters.txt'.format(args.output_dest), 'w') as txt:
            for c in sorted(maximal_clusters, key=lambda c: len(engine.clusters[c].items), reverse=True):
                cluster = engine.clusters[c]

                trimmed_pwm = trim_motif(cluster.aligned_pwms, args.info_thresh)
                chen.write('>motif{}\n'.format(c))
                for row in trimmed_pwm:
                    chen.write('{:.06f}\t{:.06f}\t{:.06f}\t{:.06f}\n'.format(*row))

                txt.write('motif{}\t{}\n'.format(c, '\t'.join([sources[item.idx][0] for item in cluster.items])))

                svg = plot_logo_stack(cluster.aligned_pwms)
                with open('{}/svgs/motif{}.svg'.format(args.output_dest, c), 'w') as f:
                    svg.writexml(f, addindent='\t', newl='\n')
