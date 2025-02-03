import numpy as np
from scipy.special import loggamma
from typing import List, Tuple, Union
import math
import itertools
from concurrent.futures import ThreadPoolExecutor
from numba import jit, float64
import numba as nb
from utils import boltzmann

@jit(float64(float64[:, :, :], float64[:], float64[:], float64, float64), nopython=True, nogil=True)
def compute_llr(aligned_pwms: np.ndarray, pc: np.ndarray, lgpc: np.ndarray, pc_sum: float, lga: float):
    npwms, width, alphabet_length = aligned_pwms.shape
    
    complement_pwm = np.sum(aligned_pwms, axis=0)
    previous_pwm = np.zeros((width, alphabet_length), dtype=np.float64)
    llr = 0.

    for i in range(npwms):
        current_pwm = aligned_pwms[i, :, :]
        complement_pwm += previous_pwm - current_pwm
        previous_pwm = current_pwm
        for j in range(width):
            current_sum = sum(current_pwm[j, :])
            complement_sum = sum(complement_pwm[j, :])
            llr += loggamma(current_sum + pc_sum) + loggamma(complement_sum + pc_sum) - \
                loggamma(current_sum + complement_sum + pc_sum) - lga
            for b in range(alphabet_length):
                current_count = current_pwm[j, b]
                complement_count = complement_pwm[j, b]
                llr += loggamma(current_count + complement_count + pc[b]) + lgpc[b] - \
                    loggamma(current_count + pc[b]) - loggamma(complement_count + pc[b])

    return llr

@jit(nb.types.Tuple((float64, float64[:, :, :], float64))(float64[:, :, :], float64[:], float64[:],
    float64, float64[:, :, :], float64[:], float64[:], float64, float64[:], float64[:], float64,
    float64, float64, float64, float64), nopython=True, nogil=True)
def compute_maximal_llr(aligned_pwms1: np.ndarray, bits1: np.ndarray, min_bits1: np.ndarray, llr1: float,
                        aligned_pwms2: np.ndarray, bits2: np.ndarray, min_bits2: np.ndarray, llr2: float,
                        pc: np.ndarray, lgpc: np.ndarray, pc_sum: float, lga: float,
                        min_information_overlap: float, max_information_overhang: float,
                        concentration: float):
    n1, width1, a = aligned_pwms1.shape
    n2, width2, a = aligned_pwms2.shape
    bits2_reverse = np.flip(bits2)
    min_bits2_reverse = np.flip(min_bits2)
    reverse_pwms2 = np.flipud(np.flip(aligned_pwms2))

    min_width = min(width1, width2)
    
    maximal_llr = -np.inf
    aligned_pwms = np.zeros((0, 0, 0), dtype=np.float64)

    for i in range(width1 + width2 - 1):
        start1 = max(i - width1 + 1, 0)
        start2 = max(width1 - i - 1, 0)
        overlap = min(i + 1, width1 + width2 - i - 1, min_width)

        info_overlap_forward = np.sum(min_bits1[start2:start2 + overlap] * \
            min_bits2[start1:start1 + overlap])
        info_overhang_forward = np.sum(bits1[:start2]) + np.sum(bits1[start2 + overlap:]) + \
            np.sum(bits2[:start1]) + np.sum(bits2[start1 + overlap:]) + \
            np.sum(np.abs(bits1[start2:start2 + overlap] - bits2[start1:start1 + overlap]))

        info_overlap_reverse = np.sum(min_bits1[start2:start2 + overlap] * \
            min_bits2_reverse[start1:start1 + overlap])
        info_overhang_reverse = np.sum(bits1[:start2]) + np.sum(bits1[start2 + overlap:]) + \
            np.sum(bits2_reverse[:start1]) + np.sum(bits2_reverse[start1 + overlap:]) + \
            np.sum(np.abs(bits1[start2:start2 + overlap] - bits2_reverse[start1:start1 + overlap]))
        
        if info_overlap_forward >= min_information_overlap and \
                info_overhang_forward <= max_information_overhang:
            combined_pwms = np.zeros((n1 + n2, width1 + max(i, width2 - 1) - min(i, width1 - 1), a),
                                     dtype=np.float64)
            combined_pwms[:n1, start1:start1 + width1, :] = aligned_pwms1
            combined_pwms[n1:, start2:start2 + width2, :] = aligned_pwms2
            potential_llr = compute_llr(combined_pwms, pc, lgpc, pc_sum, lga)
            if potential_llr > maximal_llr:
                maximal_llr = potential_llr
                aligned_pwms = combined_pwms

        if info_overlap_reverse >= min_information_overlap and \
                info_overhang_reverse <= max_information_overhang:
            combined_pwms = np.zeros((n1 + n2, width1 + max(i, width2 - 1) - min(i, width1 - 1), a),
                                     dtype=np.float64)
            combined_pwms[:n1, start1:start1 + width1, :] = aligned_pwms1
            combined_pwms[n1:, start2:start2 + width2, :] = reverse_pwms2
            potential_llr = compute_llr(combined_pwms, pc, lgpc, pc_sum, lga)
            if potential_llr > maximal_llr:
                maximal_llr = potential_llr
                aligned_pwms = combined_pwms

    scaled_llr = maximal_llr * (n1 + n2) ** concentration
    return scaled_llr, aligned_pwms, scaled_llr - llr1 - llr2

class GreedyItem:
    def __init__(self, idx: int, pwm: np.ndarray):
        self.idx = idx
        self.pwm = pwm
        self.revcomp = np.flip(pwm)
        self.width = pwm.shape[0]

class GreedyCluster:
    def __init__(self, idx: int, items: List[GreedyItem], aligned_pwms: np.ndarray,
                 llr: float, merged_from: Union[None, Tuple[int, int]]):
        self.idx = idx
        self.items = items
        self.aligned_pwms = aligned_pwms
        self.width = aligned_pwms.shape[1]
        self.llr = llr
        
        aligned_pwms_eps = aligned_pwms + 1e-20
        aligned_posterior_pwms = aligned_pwms_eps / np.sum(aligned_pwms_eps, axis=2, keepdims=True)
        min_bits = boltzmann(np.sum(aligned_posterior_pwms * \
            np.log2(aligned_posterior_pwms), axis=2) + 2., -2., axis=0)
        self.min_bits = min_bits
        
        consensus_pwm = np.sum(aligned_pwms_eps, axis=0) / \
            np.expand_dims(np.sum(aligned_pwms_eps, axis=(0, 2)), 1)
        bits = np.sum(consensus_pwm * np.log2(consensus_pwm), axis=1) + 2
        self.bits = bits
        
        self.merged_from = merged_from

class GreedyEngine:
    def __init__(self, items: List[GreedyItem], alphabet_length: int = 4, pc: np.ndarray = np.ones(4),
                 min_information_overlap: float = 8., max_information_overhang: float = 12.,
                 concentration: float = 1.):
        self.items = items
        self.alphabet_length = alphabet_length
        self.clusters = [GreedyCluster(idx, [item], item.pwm.reshape(1, *item.pwm.shape), 0., None)
                         for idx, item in enumerate(items) if item.pwm.shape[1] == alphabet_length]
        
        self.pc = np.array(pc)
        self.lgpc = loggamma(self.pc)
        self.pc_sum = np.sum(self.pc)
        self.lga = loggamma(self.pc_sum)
        
        self.min_information_overlap = min_information_overlap
        self.max_information_overhang = max_information_overhang
        self.concentration = concentration
        self.clusters_trace = [list(range(len(items)))]
        self.llr_trace = [0.]
        self.cache = {}

    def compute_llr_for_clusters(self, c1: int, c2: int):
        cluster1 = self.clusters[c1]
        aligned_pwms1 = cluster1.aligned_pwms
        bits1 = cluster1.bits
        min_bits1 = cluster1.min_bits
        llr1 = cluster1.llr

        cluster2 = self.clusters[c2]
        aligned_pwms2 = cluster2.aligned_pwms
        bits2 = cluster2.bits
        min_bits2 = cluster2.min_bits
        llr2 = cluster2.llr

        return c1, c2, compute_maximal_llr(aligned_pwms1, bits1, min_bits1, llr1, aligned_pwms2, bits2,
                                           min_bits2, llr2, self.pc, self.lgpc, self.pc_sum, self.lga,
                                           self.min_information_overlap, self.max_information_overhang,
                                           self.concentration)

    def one_iteration(self, n_processes: Union[None, int] = None):
        current_clusters = list(self.clusters_trace[-1])
        nclusters = len(current_clusters)
        all_combos = set(itertools.combinations(current_clusters, 2))

        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            for c1, c2, results in executor.map(self.compute_llr_for_clusters,
                                                *zip(*(all_combos - set(self.cache)))):
                self.cache[(c1, c2)] = results
        c1, c2 = max(self.cache, key=lambda k: self.cache[k][2])
        llr, aligned_pwms, _ = self.cache[(c1, c2)]
    
        if math.isinf(llr):
            return False
            
        cluster1 = self.clusters[c1]
        cluster2 = self.clusters[c2]
        cluster_idx = len(self.clusters)
        cluster = GreedyCluster(cluster_idx, cluster1.items + cluster2.items, aligned_pwms, llr,
                                merged_from=(c1, c2))
        self.clusters.append(cluster)

        current_clusters.remove(c1)
        current_clusters.remove(c2)
        current_clusters.append(cluster_idx)
        self.clusters_trace.append(current_clusters)
        self.llr_trace.append(sum(self.clusters[c].llr for c in current_clusters))
        
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
