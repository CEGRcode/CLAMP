import numpy as np
import argparse
from CLAMP.parse_tf import parse_tf

def write_pfm(motif_id, pwm, nsites, forward_offset, reverse_offset, rc, out_fh,
              rng=np.random.default_rng()):
    out_fh.write('AC\t{}\n'.format(motif_id))
    out_fh.write('XX\n')
    out_fh.write('ID\t{}\n'.format(motif_id))
    out_fh.write('PO	A	C	G	T\n')
    for j in range(pwm.shape[0]):
        out_fh.write('{:02d}\t{}\t{}\t{}\t{}\n'.format(j + 1, *rng.multinomial(nsites, pwm[j, :])))
    out_fh.write('XX\n')
    out_fh.write('FO\t{}\n'.format(forward_offset))
    out_fh.write('RO\t{}\n'.format(reverse_offset))
    out_fh.write('RC\t{}\n'.format(int(rc)))
    out_fh.write('XX\n//\n')

def simulate_aperiodic_motifs(base_motif_id, pwm, out_fh, exp_n=5., min_width=4, min_sites=20,
                              max_sites=100, rng=np.random.default_rng()):
    entr = pwm.copy()
    entr[entr > 0] *= np.log2(entr[entr > 0])
    p = -np.sum(entr, axis=1) / 2 + 1e-10
    forward_weights = np.cumsum(np.log(np.insert(p, 0, 1.))) + np.log1p(np.append(-p, 0.))
    reverse_weights = np.cumsum(np.log(np.insert(np.flip(p), 0, 1.))) + \
        np.log1p(np.append(-np.flip(p), 0.))
    
    n = rng.poisson(exp_n)
    rc = rng.binomial(1, .5, size=n).astype(bool)
    nsites = rng.integers(min_sites, max_sites + 1, size=n)

    i = 0
    while i < n:
        motif_id = '{}-{}'.format(base_motif_id, i + 1)

        forward_offset = np.argmax(rng.gumbel(forward_weights))
        reverse_offset = np.argmax(rng.gumbel(reverse_weights))

        trunc_pwm = pwm[forward_offset:-reverse_offset, :] if reverse_offset > 0 \
            else pwm[forward_offset:, :]
        
        if rc[i]:
            trunc_pwm = np.flip(trunc_pwm)
        trunc_pwm /= np.sum(trunc_pwm, axis=1, keepdims=True)
        if trunc_pwm.shape[0] < min_width:
            continue
        
        write_pfm(motif_id, trunc_pwm, nsites[i], forward_offset, reverse_offset, rc[i], out_fh, rng)
        i += 1

def simulate_periodic_motifs(base_motif_id, pwm, out_fh, exp_n=5., min_width=4, min_sites=20,
                             max_sites=100, padding_p=.5, bg_alpha=100., rng=np.random.default_rng()):
    entr = pwm.copy()
    entr[entr > 0] *= np.log2(entr[entr > 0])
    p = -np.sum(entr, axis=1) / 2 + 1e-10
    forward_weights = np.cumsum(np.log(np.insert(p, 0, 1.))) + np.log1p(np.append(-p, 0.))
    reverse_weights = np.cumsum(np.log(np.insert(np.flip(p), 0, 1.))) + \
        np.log1p(np.append(-np.flip(p), 0.))
    
    n = rng.poisson(exp_n)
    rc = rng.binomial(1, .5, size=n).astype(bool)
    offset_dir = rng.binomial(1, .5, size=n).astype(bool)
    nsites = rng.integers(min_sites, max_sites + 1, size=n)

    i = 0
    while i < n:
        motif_id = '{}-{}'.format(base_motif_id, i + 1)

        if offset_dir[i]:
            forward_offset = np.argmax(rng.gumbel(forward_weights))
            reverse_offset = -rng.geometric(padding_p) + 1
        else:
            forward_offset = -rng.geometric(padding_p) + 1
            reverse_offset = np.argmax(rng.gumbel(reverse_weights))
            
        trunc_pwm = np.empty((pwm.shape[0] - forward_offset - reverse_offset, 4), dtype=np.float64)
        if offset_dir[i]:
            if reverse_offset == 0:
                trunc_pwm[:, :] = pwm[forward_offset:, :]
            else:
                trunc_pwm[:reverse_offset, :] = pwm[forward_offset:, :]
                for j in range(reverse_offset, 0):
                    trunc_pwm[j, :] = rng.dirichlet([bg_alpha] * 4)
        else:
            if reverse_offset == 0:
                trunc_pwm[-forward_offset:, :] = pwm
            else:
                trunc_pwm[-forward_offset:, :] = pwm[:-reverse_offset, :]
            for j in range(-forward_offset):
                trunc_pwm[j, :] = rng.dirichlet([bg_alpha] * 4)
                    
        if rc[i]:
            trunc_pwm = np.flip(trunc_pwm)
        trunc_pwm /= np.sum(trunc_pwm, axis=1, keepdims=True)
        if trunc_pwm.shape[0] < min_width:
            continue

        write_pfm(motif_id, trunc_pwm, nsites[i], forward_offset, reverse_offset, rc[i], out_fh, rng)
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate motifs from base pwms')
    parser.add_argument('base_pwms', type=str)
    parser.add_argument('--exp-n', type=float, default=5.)
    parser.add_argument('--min-width', type=int, default=4)
    parser.add_argument('--min-sites', type=int, default=20)
    parser.add_argument('--max-sites', type=int, default=100)
    parser.add_argument('--padding-p', type=float, default=.5)
    parser.add_argument('--bg-alpha', type=float, default=100.)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', type=str, default='motifs.transfac')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base_pwms, metadata = parse_tf(args.base_pwms)
    with open(args.out, 'w') as f:
        for motif_id, pwm in base_pwms.items():
            if metadata[motif_id]['PD'] == 0:
                simulate_aperiodic_motifs(motif_id, pwm, f, args.exp_n, args.min_width,
                                           args.min_sites, args.max_sites, rng)
            else:
                simulate_periodic_motifs(motif_id, pwm, f, args.exp_n, args.min_width,
                                         args.min_sites, args.max_sites, args.padding_p,
                                         args.bg_alpha, rng)
