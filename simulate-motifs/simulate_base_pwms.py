import numpy as np
from scipy.special import digamma
import argparse

a = np.logspace(-10, 3, 1001)
h_hat = digamma(4 * a + 1) - digamma(a + 1)

def simulate_base_pwms_default(entr_dist, n=100, min_width=6, max_width=30,
                               min_logits=5., bg_alpha=100., rng=np.random.default_rng()):
    pwms = []
    
    widths = rng.integers(min_width, max_width + 1, size=n)
    left_pad = (max_width - widths) // 2
    right_pad = max_width - widths - left_pad
    for i in range(n):
        entr = rng.choice(entr_dist, size=widths[i])
        if sum(np.log(4) - entr) < min_logits:
            entr *= (widths[i] * np.log(4) - min_logits) / np.sum(entr)
        alpha = np.insert(np.interp(entr, h_hat, a), [0] * left_pad[i] + [widths[i]] * right_pad[i], bg_alpha)
        
        pwms.append(np.stack([rng.dirichlet([alpha[j]] * 4) for j in range(max_width)]))
    return pwms

def simulate_base_pwms_palindromic(entr_dist, n=100, min_width=6, max_width=30,
                                   min_logits=5., bg_alpha=100., rng=np.random.default_rng()):
    pwms = []
    
    widths = rng.integers(min_width, max_width + 1, size=n)
    left_pad = (max_width - widths) // 2
    for i in range(n):
        halfwidth = int(np.ceil(widths[i] / 2))
        entr = rng.choice(entr_dist, size=halfwidth)
        entr_ = np.concatenate([entr, np.flip(entr)])[:widths[i]]
        if sum(np.log(4) - entr_) < min_logits:
            entr *= (widths[i] * np.log(4) - min_logits) / np.sum(entr_)
        alpha = np.interp(entr, h_hat, a)
        
        half_pwm = np.empty((halfwidth, 4))
        for j in range(halfwidth):
            half_pwm[j, :] = rng.dirichlet([alpha[j]] * 4)
        pwm = np.zeros((max_width, 4), dtype=np.float64)
        pwm[left_pad[i]:left_pad[i] + widths[i], :] = np.concatenate([half_pwm, np.flip(half_pwm)])[:widths[i], :]
        for j in range(left_pad[i]):
            pwm[j, :] = rng.dirichlet([bg_alpha] * 4)
        for j in range(left_pad[i] + widths[i], max_width):
            pwm[j, :] = rng.dirichlet([bg_alpha] * 4)
        pwms.append(pwm)
    return pwms

def simulate_base_pwms_periodic(entr_dist, n=100, min_width=6, max_width=30,
                                min_period_width=4, max_period_width=6, min_logits=2.,
                                min_period_stability=10., max_period_stability=100.,
                                rng=np.random.default_rng()):
    pwms = []
    
    period_widths = rng.integers(min_period_width, max_period_width + 1, size=n)
    n_repeats = rng.integers(np.maximum(2, min_width // period_widths), max_width // period_widths)
    n_before = rng.integers(0, n_repeats)
    stability_weights = 1 / np.arange(min_period_stability, max_period_stability + 1)
    stability_weights /= np.sum(stability_weights)
    period_stability = rng.choice(np.arange(min_period_stability, max_period_stability + 1),
                                  p=stability_weights, size=n)
    for i in range(n):
        entr = rng.choice(entr_dist, size=period_widths[i])
        while sum(np.log(4) - entr) < min_logits:
            entr = rng.choice(entr_dist, size=period_widths[i])
        alpha = np.interp(entr, h_hat, a)
        pc_mat = np.empty((period_widths[i], 4))
        for j in range(period_widths[i]):
            p = rng.dirichlet([alpha[j]] * 4)
            pc_mat[j, :] = rng.multinomial(period_stability[i], p)

        pwm = np.empty((period_widths[i] * n_repeats[i], 4))
        for j in range(period_widths[i] * n_repeats[i]):
            pc = abs(j // period_widths[i] - n_before[i])
            mat_idx = j % period_widths[i]
            pwm[j, :] = rng.dirichlet(pc_mat[mat_idx, :] + pc)
        pwms.append(pwm)
    return pwms, period_widths

def write_pwm(motif_id, pwm, out_fh, period=None):
    if period is None:
        period = 0
    out_fh.write('AC\t{}\n'.format(motif_id))
    out_fh.write('XX\n')
    out_fh.write('ID\t{}\n'.format(motif_id))
    out_fh.write('PO	A	C	G	T\n')
    for i in range(pwm.shape[0]):
        out_fh.write('{:02d}\t{:06f}\t{:06f}\t{:06f}\t{:06f}\n'.format(i + 1, *pwm[i, :]))
    out_fh.write('XX\n')
    out_fh.write('PD\t{}\n'.format(period))
    out_fh.write('XX\n//\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate base PWMs')
    parser.add_argument('--entr-dist', type=str, default=None)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--a-default', type=float, default=9.)
    parser.add_argument('--a-palindromic', type=float, default=2.)
    parser.add_argument('--a-periodic', type=float, default=2.)
    parser.add_argument('--min-width', type=int, default=6)
    parser.add_argument('--max-width', type=int, default=30)
    parser.add_argument('--min-logits', type=float, default=5.)
    parser.add_argument('--bg-alpha', type=float, default=100.)
    parser.add_argument('--min-period-width', type=int, default=4)
    parser.add_argument('--max-period-width', type=int, default=6)
    parser.add_argument('--min-period-logits', type=float, default=2.)
    parser.add_argument('--min-period-stability', type=float, default=10.)
    parser.add_argument('--max-period-stability', type=float, default=100.)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', type=str, default='base_pwms.transfac')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    entr_dist = np.loadtxt(args.entr_dist)
    type_props = rng.dirichlet([args.a_default, args.a_palindromic, args.a_periodic])
    n_def, n_pal, n_per = rng.multinomial(args.n, type_props)
    
    i = 1
    with open(args.out, 'w') as f:
        for pwm in simulate_base_pwms_default(entr_dist, n=n_def, min_width=args.min_width,
            max_width=args.max_width, min_logits=args.min_logits, bg_alpha=args.bg_alpha, rng=rng):

            write_pwm('motif{}'.format(i), pwm, f)
            i += 1

        for pwm in simulate_base_pwms_palindromic(entr_dist, n=n_pal, min_width=args.min_width,
            max_width=args.max_width, min_logits=args.min_logits, bg_alpha=args.bg_alpha, rng=rng):

            write_pwm('motif{}'.format(i), pwm, f)
            i += 1
        
        for pwm, period in zip(*simulate_base_pwms_periodic(entr_dist, n=n_per,
            min_width=args.min_width, max_width=args.max_width,
            min_period_width=args.min_period_width, max_period_width=args.max_period_width,
            min_logits=args.min_period_logits, min_period_stability=args.min_period_stability,
            max_period_stability=args.max_period_stability, rng=rng)):

            write_pwm('motif{}'.format(i), pwm, f, period=period)
            i += 1
            