import os
import json
from collections import namedtuple
import numpy as np
import xml.dom.minidom as dom
from utils import trim_motif

def write_aligned_transfac(cluster, filename):
    with open(filename, 'w') as f:
        for i in range(len(cluster.items)):
            motif_id = cluster.items[i].source[0]
            f.write('AC\t{}\n'.format(motif_id))
            f.write('XX\n')
            f.write('ID\t{}\n'.format(motif_id))
            f.write('PO\tA\tC\tG\tT\n')
            pfm = cluster.aligned_pfms[i, :, :]
            for j in range(pfm.shape[0]):
                f.write('{:02d}\t{:06f}\t{:06f}\t{:06f}\t{:06f}\n'.format(j + 1, *pfm[j, :]))
            f.write('XX\n//\n')
            
def write_consensus_transfac(cluster, filename, info_thresh=.5):
    c = cluster.idx
    trimmed_pfm, _, _, _ = trim_motif(cluster.aligned_pfms, info_thresh)
    with open(filename, 'w') as f:
        f.write('AC\t{}\n'.format('cluster{}'.format(c)))
        f.write('XX\n')
        f.write('ID\t{}\n'.format('cluster{}'.format(c)))
        f.write('PO\tA\tC\tG\tT\n')
        for j in range(trimmed_pfm.shape[0]):
            f.write('{:02d}\t{:06f}\t{:06f}\t{:06f}\t{:06f}\n'.format(j + 1, *trimmed_pfm[j, :]))
        f.write('XX\n//\n')

def write_consensus_meme(cluster, filename, info_thresh=.5):
    c = cluster.idx
    trimmed_pfm, _, _, _ = trim_motif(cluster.aligned_pfms, info_thresh)
    with open(filename, 'w') as f:
        f.write('MEME version 5\n')
        f.write('ALPHABET= ACGT\n')
        f.write('MOTIF {0} {0}\n'.format('cluster{}'.format(c)))
        f.write('letter-probability matrix: alength= 4 w= {} nsites= {} E= 0\n'.format(
            trimmed_pfm.shape[0], sum(item.source[1] for item in cluster.items)))
        for j in range(trimmed_pfm.shape[0]):
            f.write('{:06f} {:06f} {:06f} {:06f}\n'.format(*(trimmed_pfm[j, :] / np.sum(trimmed_pfm[j, :]))))

clamp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open('{}/logo_symbols/glyphs.json'.format(clamp_dir), 'r') as f:
    glyph_data = json.load(f)
with open('{}/logo_symbols/symbol_library.json'.format(clamp_dir), 'r') as f:
    symbol_library = json.load(f)

ColoredGlyph = namedtuple('ColoredGlyph', ['path', 'color'])

class DNASymbol:
    max_bits = symbol_library['DNA']['max_bits']
    DNA_alphabet = tuple(ColoredGlyph(path=glyph_data[s['name']]['path'], color=s['color']) \
                         for s in symbol_library['DNA']['symbols'])

    @classmethod
    def get_symbol(cls, i):
        return cls.DNA_alphabet[i]

class RNASymbol:
    max_bits = symbol_library['RNA']['max_bits']
    RNA_alphabet = tuple(ColoredGlyph(path=glyph_data[s['name']]['path'], color=s['color']) \
                         for s in symbol_library['RNA']['symbols'])
    
    @classmethod
    def get_symbol(cls, i):
        return cls.RNA_alphabet[i]

class AASymbol:
    max_bits = symbol_library['AA']['max_bits']
    AA_alphabet = tuple(ColoredGlyph(path=glyph_data[s['name']]['path'], color=s['color']) \
                        for s in symbol_library['AA']['symbols'])
    
    @classmethod
    def get_symbol(cls, i):
        return cls.AA_alphabet[i]

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
