import numpy as np
import xml.dom.minidom as dom
from typing import Union
from scipy.stats import pearsonr

def highest_n_info_sum(pfm, n=4, w=3):
    '''
    Sum of the information content of the highest n windows of width w in a PFM
    '''
    pwm = pfm / np.sum(pfm, axis=1, keepdims=True)
    logpwm = np.array(pwm)
    logpwm[pwm > 0.] = np.log2(pwm[pwm > 0.])
    bits = np.sum(pwm * logpwm, axis=1) + 2.
    val = sum(bits[:w]) / w
    mean_bits = [val]
    for i in range(w, len(bits)):
        val += (bits[i] - bits[i - w]) / w
        mean_bits.append(val)
    return sum(sorted(mean_bits, reverse=True)[:n])

def check_periodicity(pfm, p=1):
    '''
    Periodicity score for a PFM based on the weighted Pearson correlation between
    positions separated by p bases, weighted by the information content of the bases
    '''
    pwm = pfm / np.sum(pfm, axis=1, keepdims=True)
    logpwm = np.array(pwm)
    logpwm[pwm > 0.] = np.log2(pwm[pwm > 0.])
    bits = np.sum(pwm * logpwm, axis=1) + 2.
    w = pfm.shape[0]
    per = []
    total_bit_prod = 0
    for i in range(p, w, p):
        per = np.concatenate([per, pearsonr(pfm[i:, :].T, pfm[:-i, :].T)[0] * bits[i:] * bits[:-i]])
        total_bit_prod += np.sum(bits[i:] * bits[:-i])
    per[np.isnan(per)] = 0.
    return np.sum(per) / total_bit_prod

def filter_motifs(items, nsites_thresh=10, evalue_thresh=.01, info_score_thresh=5., periodicity1_thresh=.6,
                  periodicity2_thresh=.75, periodicity3_thresh=.75):
    '''
    Filters a list of GreedyItems based on various QC thresholds
    '''
    new_items = []
    for item in items:
        nsites = item.source[1]
        evalue = item.source[2]
        pfm = item.pfm
        info_score = highest_n_info_sum(pfm)
        periodicity1 = check_periodicity(pfm, p=1)
        periodicity2 = check_periodicity(pfm, p=2)
        periodicity3 = check_periodicity(pfm, p=3)
        if nsites >= nsites_thresh and evalue <= evalue_thresh and info_score >= info_score_thresh \
           and periodicity1 <= periodicity1_thresh and periodicity2 <= periodicity2_thresh \
           and periodicity3 <= periodicity3_thresh:
            new_items.append(item)
    return new_items

def boltzmann(arr: np.ndarray, alpha: float, axis: Union[None, int, tuple] = None):
    '''
    Boltzmann operator as described by https://en.wikipedia.org/wiki/Smooth_maximum#Boltzmann_operator
    '''
    return np.sum(arr * np.exp(alpha * arr), axis=axis) / np.sum(np.exp(alpha * arr), axis=axis)
    
def trim_motif(aligned_pfms, info_thresh=.5, w=2):
    '''
    Trims bases from the start and end of a PFM stack with information content below info_thresh
    '''
    pfm = np.sum(aligned_pfms, axis=0)

    # If there is only one motif, return the original PFM
    if aligned_pfms.shape[0] == 1:
        return pfm, 0, pfm.shape[0], False

    # Calculate the posterior probability of each base (1 pseudocount)
    pwm = pfm / np.sum(pfm, axis=1, keepdims=True)
    pwm[pwm == 0.] = 1.
    # Calculate the information content of each base
    bits = np.sum(pwm * np.log2(pwm), axis=1) + np.log2(aligned_pfms.shape[2])
    val = sum(bits[:w]) / w
    mean_bits = np.zeros(aligned_pfms.shape[1] - w + 1, dtype=np.float64)
    mean_bits[0] = val
    for i in range(w, len(bits)):
        val += (bits[i] - bits[i - w]) / w
        mean_bits[i - w + 1] = val
    # Trim the PFM to only include bases with information content above the threshold
    informative_bits = np.flatnonzero(mean_bits > info_thresh)
    if len(informative_bits) <= 1:
        return pfm, 0, pfm.shape[0], False
    start = informative_bits[0]
    end = informative_bits[-1] + w
    return pfm[start:end, :], start, pfm.shape[0] - end, True

# Classes for DNA symbols in SVG format
class DNASymbol:
    path = None
    color = None
    max_bits = 2
    DNA_alphabet = ()

    @classmethod
    def get_symbol(cls, i):
        return cls.DNA_alphabet[i]

class DNA_A(DNASymbol):
    path = 'M 0 100 L 33 0 L 66 0 L 100 100 L 75 100 L 66 75 L 33 75 L 25 100 L 0 100 M 41 55 L 58 55 L 50 25 L 41 55'
    color = '#FF0000'

class DNA_C(DNASymbol):
    path = 'M 100 28 C 100 -13 0 -13 0 50 C 0 113 100 113 100 72 L 75 72 C 75 90 30 90 30 50 C 30 10 75 10 75 28 Z'
    color = '#0000FF'

class DNA_G(DNASymbol):
    path = 'M 100 28 C 100 -13 0 -13 0 50 C 0 113 100 113 100 72 L 100 48 L 55 48 L 55 72 L 75 72 C 75 90 30 90 30 50 C 30 10 75 5 75 28 Z'
    color = '#FFA500'

class DNA_T(DNASymbol):
    path = 'M 0 0 L 0 20 L 35 20 L 35 100 L 65 100 L 65 20 L 100 20 L 100 0 Z'
    color = '#228B22'

DNASymbol.DNA_alphabet = (DNA_A, DNA_C, DNA_G, DNA_T)

# TODO: Add RNA and protein symbols

def plot_logo_stack(aligned_pfms, symbol=DNASymbol, glyph_width=100, stack_height=200):
    '''
    Plots a stack of logos for a stack of aligned PFMs and outputs it as an SVG
    To save the SVG:
        doc = plot_logo_stack(aligned_pfms)
        with open(filename, 'w') as f:
            doc.writexml(f, addindent='\t', newl='\n')
    '''
    height, width, _ = aligned_pfms.shape

    document = dom.Document()
    svg = document.appendChild(document.createElement('svg'))
    svg.setAttribute('baseProfile', 'full')
    svg.setAttribute('version', '1.1')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('viewBox', '0 0 {} {}'.format(width * glyph_width, height * stack_height))

    logo_stack = svg.appendChild(document.createElement('g'))

    for y, pfm in enumerate(aligned_pfms):
        row = logo_stack.appendChild(document.createElement('g'))
        row.setAttribute('transform', 'translate(0 {})'.format(y * stack_height))
        for i, pwv in enumerate(pfm):
            n = np.sum(pwv)
            if n == 0:
                continue
            vec = pwv / n
            vec[vec > 0] *= np.log2(vec[vec > 0])
            bits = sum(vec, symbol.max_bits)
            heights = pwv / n * bits / symbol.max_bits * stack_height
            idx = np.argsort(pwv)
    
            stack = row.appendChild(document.createElement('g'))
            stack.setAttribute('transform', 'translate({} 0)'.format(i * glyph_width))
    
            y_offset = 0
            for j in idx:
                if heights[j] == 0:
                    continue
                base = symbol.get_symbol(j)
                y_offset += heights[j]
                glyph = stack.appendChild(document.createElement('path'))
                glyph.setAttribute('d', base.path)
                glyph.setAttribute('fill', base.color)
                glyph.setAttribute('transform', 'matrix({} 0 0 {} 0 {})'.format(glyph_width / 100., heights[j] / 100., stack_height - y_offset))

    return svg