import numpy as np
from scipy.special import loggamma
from typing import List, Tuple, Union
from math import isinf
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from numba import jit, int64, float64, boolean
import numba as nb
from ctypes import CFUNCTYPE, c_double
from numba.extending import overload, get_cython_function_address
from utils import boltzmann

# Overload scipy loggamma to use in numba jit functions
loggamma_c = CFUNCTYPE(c_double, c_double)(
    get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1loggamma')
)
def loggamma_kernel(*args):
    if args == (float64,):
        return lambda *args: loggamma_c(*args)
overload(loggamma)(loggamma_kernel)

@jit(float64(float64[:, :, :], float64[:], float64[:], float64, float64), nopython=True, nogil=True)
def compute_llr(aligned_pfms: np.ndarray, pc: np.ndarray, lgpc: np.ndarray, pc_sum: float, lga: float):
    npfms, width, alphabet_length = aligned_pfms.shape
    
    complement_pfm = np.sum(aligned_pfms, axis=0)
    previous_pfm = np.zeros((width, alphabet_length), dtype=np.float64)
    llr = 0.

    for i in range(npfms):
        current_pfm = aligned_pfms[i, :, :]
        complement_pfm += previous_pfm - current_pfm
        previous_pfm = current_pfm
        for j in range(width):
            current_sum = np.sum(current_pfm[j, :])
            complement_sum = np.sum(complement_pfm[j, :])
            llr += loggamma(current_sum + pc_sum) + loggamma(complement_sum + pc_sum) - \
                loggamma(current_sum + complement_sum + pc_sum) - lga
            for b in range(alphabet_length):
                current_count = current_pfm[j, b]
                complement_count = complement_pfm[j, b]
                llr += loggamma(current_count + complement_count + pc[b]) + lgpc[b] - \
                    loggamma(current_count + pc[b]) - loggamma(complement_count + pc[b])

    return llr

@jit(nb.types.Tuple((float64[:, :, :], float64, float64, float64, float64, int64, int64, int64, int64, boolean))(
    float64[:, :, :], float64[:], float64[:], float64, float64[:, :, :], float64[:], float64[:],
    float64, float64[:], float64[:], float64, float64, int64, float64, float64, float64),
        nopython=True, nogil=True)
def compute_maximal_llr(aligned_pfms1: np.ndarray, bits1: np.ndarray, min_bits1: np.ndarray, llr1: float,
                        aligned_pfms2: np.ndarray, bits2: np.ndarray, min_bits2: np.ndarray, llr2: float,
                        pc: np.ndarray, lgpc: np.ndarray, pc_sum: float, lga: float,
                        min_base_overlap: int, min_information_overlap: float,
                        max_information_overhang: float, concentration: float):
    n1, width1, a = aligned_pfms1.shape
    n2, width2, a = aligned_pfms2.shape
    bits2_reverse = np.flip(bits2)
    min_bits2_reverse = np.flip(min_bits2)
    reverse_pfms2 = np.flipud(np.flip(aligned_pfms2))

    min_width = min(width1, width2)
    
    maximal_llr = -np.inf
    aligned_pfms = np.zeros((0, 0, 0), dtype=np.float64)
    left_offset1 = 0
    right_offset1 = 0
    left_offset2 = 0
    right_offset2 = 0
    rc = False
    for i in range(min_base_overlap - 1, width1 + width2 - min_base_overlap):
        start1 = max(i - width1 + 1, 0)
        start2 = max(width1 - i - 1, 0)
        overlap = min(i + 1, width1 + width2 - i - 1, min_width)
        combined_width = width1 + max(i, width2 - 1) - min(i, width1 - 1)

        info_overlap_forward = np.sum(np.minimum(min_bits1[start2:start2 + overlap],
            min_bits2[start1:start1 + overlap]))
        info_overhang_forward = np.sum(bits1[:start2]) + np.sum(bits1[start2 + overlap:]) + \
            np.sum(bits2[:start1]) + np.sum(bits2[start1 + overlap:]) + \
            np.sum(np.abs(bits1[start2:start2 + overlap] - bits2[start1:start1 + overlap]))

        info_overlap_reverse = np.sum(np.minimum(min_bits1[start2:start2 + overlap],
            min_bits2_reverse[start1:start1 + overlap]))
        info_overhang_reverse = np.sum(bits1[:start2]) + np.sum(bits1[start2 + overlap:]) + \
            np.sum(bits2_reverse[:start1]) + np.sum(bits2_reverse[start1 + overlap:]) + \
            np.sum(np.abs(bits1[start2:start2 + overlap] - bits2_reverse[start1:start1 + overlap]))
        
        if info_overlap_forward >= min_information_overlap and \
                info_overhang_forward <= max_information_overhang:
            combined_pfms = np.zeros((n1 + n2, combined_width, a),
                                     dtype=np.float64)
            combined_pfms[:n1, start1:start1 + width1, :] = aligned_pfms1
            combined_pfms[n1:, start2:start2 + width2, :] = aligned_pfms2
            potential_llr = compute_llr(combined_pfms, pc, lgpc, pc_sum, lga)# + info_overlap_forward - info_overhang_forward
            if potential_llr > maximal_llr:
                maximal_llr = potential_llr
                aligned_pfms = combined_pfms
                left_offset1 = start1
                right_offset1 = combined_width - width1 - start1
                left_offset2 = start2
                right_offset2 = combined_width - width2 - start2
                rc = False

        if info_overlap_reverse >= min_information_overlap and \
                info_overhang_reverse <= max_information_overhang:
            combined_pfms = np.zeros((n1 + n2, combined_width, a),
                                     dtype=np.float64)
            combined_pfms[:n1, start1:start1 + width1, :] = aligned_pfms1
            combined_pfms[n1:, start2:start2 + width2, :] = reverse_pfms2
            potential_llr = compute_llr(combined_pfms, pc, lgpc, pc_sum, lga)# + info_overlap_reverse - info_overhang_reverse
            if potential_llr > maximal_llr:
                maximal_llr = potential_llr
                aligned_pfms = combined_pfms
                left_offset1 = start1
                right_offset1 = combined_width - width1 - start1
                left_offset2 = start2
                right_offset2 = combined_width - width2 - start2
                rc = True

    scaled_llr = maximal_llr * (n1 + n2) ** concentration
    scaled_llr1 = llr1 * n1 ** concentration
    scaled_llr2 = llr2 * n2 ** concentration
    return aligned_pfms, maximal_llr, maximal_llr - llr1 - llr2, scaled_llr, scaled_llr - scaled_llr1 - scaled_llr2, left_offset1, right_offset1, left_offset2, right_offset2, rc

class GreedyItem:
    def __init__(self, idx: int, pfm: np.ndarray, source: str, sites: frozenset):
        self.idx = idx
        self.pfm = pfm
        self.revcomp = np.flip(pfm)
        self.width = pfm.shape[0]
        self.source = source
        self.sites = sites

class GreedyCluster:
    def __init__(self, idx: int, items: List[GreedyItem], aligned_pfms: np.ndarray,
                 llr: float, sites: dict, merged_from: Union[None, Tuple[int, int]]):
        self.idx = idx
        self.items = items
        self.aligned_pfms = aligned_pfms
        self.width = aligned_pfms.shape[1]
        self.llr = llr
        self.sites = sites
        
        aligned_pfms_eps = aligned_pfms + 1e-20
        aligned_posterior_pfms = aligned_pfms_eps / np.sum(aligned_pfms_eps, axis=2, keepdims=True)
        min_bits = boltzmann(np.sum(aligned_posterior_pfms * \
            np.log2(aligned_posterior_pfms), axis=2) + 2., -2., axis=0)
        self.min_bits = min_bits
        
        consensus_pfm = np.sum(aligned_pfms_eps, axis=0) / \
            np.expand_dims(np.sum(aligned_pfms_eps, axis=(0, 2)), 1)
        bits = np.sum(consensus_pfm * np.log2(consensus_pfm), axis=1) + 2
        self.bits = bits
        
        self.merged_from = merged_from

class GreedyEngine:
    def __init__(self, items: List[GreedyItem], pc: np.ndarray = np.ones(4),
                 min_base_overlap: int = 4, min_information_overlap: float = 8.,
                 max_information_overhang: float = 12., concentration: float = 1.):
        self.items = items
        self.clusters = [GreedyCluster(idx, [item], item.pfm.reshape(1, *item.pfm.shape), 0.,
                                       {site: {item.source} for site in item.sites}, None)
                         for idx, item in enumerate(items) if item.pfm.shape[1] == len(pc)]
        
        self.pc = np.array(pc)
        self.lgpc = loggamma(self.pc)
        self.pc_sum = np.sum(self.pc)
        self.lga = loggamma(self.pc_sum)

        self.min_base_overlap = min_base_overlap
        self.min_information_overlap = min_information_overlap
        self.max_information_overhang = max_information_overhang
        self.concentration = concentration
        self.clusters_trace = [list(range(len(items)))]
        self.llr_trace = [0.]
        self.cache = {}

    def compute_llr_for_clusters(self, c1: int, c2: int):
        cluster1 = self.clusters[c1]
        aligned_pfms1 = cluster1.aligned_pfms
        bits1 = cluster1.bits
        min_bits1 = cluster1.min_bits
        llr1 = cluster1.llr

        cluster2 = self.clusters[c2]
        aligned_pfms2 = cluster2.aligned_pfms
        bits2 = cluster2.bits
        min_bits2 = cluster2.min_bits
        llr2 = cluster2.llr

        return c1, c2, compute_maximal_llr(aligned_pfms1, bits1, min_bits1, llr1, aligned_pfms2, bits2,
                                           min_bits2, llr2, self.pc, self.lgpc, self.pc_sum, self.lga,
                                           self.min_base_overlap, self.min_information_overlap,
                                           self.max_information_overhang, self.concentration)

    def one_iteration(self, n_processes: Union[None, int] = None):
        current_clusters = list(self.clusters_trace[-1])
        all_combos = set(combinations(current_clusters, 2))

        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            for c1, c2, results in executor.map(self.compute_llr_for_clusters,
                                                *zip(*(all_combos - set(self.cache)))):
                self.cache[(c1, c2)] = results
        # c1, c2 = max(self.cache, key=lambda k: self.cache[k][4])
        c1, c2 = max(self.cache, key=lambda k: self.cache[k][4] if self.cache[k][2] >= 0 else -np.inf)
        aligned_pfms, llr, _, scaled_llr, _, left_offset1, right_offset1, left_offset2, right_offset2, rc = self.cache[(c1, c2)]
    
        if isinf(llr):
            return False
            
        cluster1 = self.clusters[c1]
        cluster2 = self.clusters[c2]
        cluster_idx = len(self.clusters)
        
        sites = {}
        for site in cluster1.sites:
            chrom, start, stop, strand = site
            if strand == '+':
                expanded_site = (chrom, start - left_offset1, stop + right_offset1, '+')
            else:
                expanded_site = (chrom, start - right_offset1, stop + left_offset1, '-')
            sites.setdefault(expanded_site, set())
            sites[expanded_site].update(cluster1.sites[site])
        for site in cluster2.sites:
            chrom, start, stop, strand = site
            if (strand == '+' and rc) or (strand == '-' and not rc):
                expanded_site = (chrom, start - right_offset2, stop + left_offset2, '-')
            else:
                expanded_site = (chrom, start - left_offset2, stop + right_offset2, '+')
            sites.setdefault(expanded_site, set())
            sites[expanded_site].update(cluster2.sites[site])
                
        cluster = GreedyCluster(cluster_idx, cluster1.items + cluster2.items, aligned_pfms, llr,
                                sites, merged_from=(c1, c2))
        self.clusters.append(cluster)

        current_clusters.remove(c1)
        current_clusters.remove(c2)
        current_clusters.append(cluster_idx)
        self.clusters_trace.append(current_clusters)
        self.llr_trace.append(sum(self.clusters[c].llr * len(self.clusters[c].items) ** self.concentration for c in current_clusters))
        
        self.cache.pop((c1, c2))
        for c in current_clusters:
            self.cache.pop((c, c1), None)
            self.cache.pop((c1, c), None)
            self.cache.pop((c, c2), None)
            self.cache.pop((c2, c), None)
            
        return True

    def cluster_motifs(self, n_processes: Union[None, int] = None):
        n_iter = len(self.clusters_trace[-1]) - 1
        for i in range(n_iter):
            print('{}/{}      '.format(i + 1, n_iter), end='\r')
            if not self.one_iteration(n_processes=n_processes):
                print('No more valid merges... done')
                break
        print()
