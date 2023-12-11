import tqdm
import matplotlib.pyplot as plt
import string
import random
import operator
import numpy as np
import numba
from numba import njit

# Silence numba warnings
import warnings
warnings.filterwarnings('ignore', category=numba.NumbaPendingDeprecationWarning)

# Moved out for numba compatability

class HDC:
    SIZE = 10_000
    def __init__(self):
        pass

    @staticmethod
    def rand_vec() -> np.ndarray:
        """generate atomic hypervector with size HDC.SIZE"""
        return np.random.uniform(0.0, 1.0, HDC.SIZE) < 0.5

    @staticmethod
    def dist(x1, x2) -> int:
        return np.mean(x1 != x2)

    @staticmethod
    def bind(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """bind two hypervectors together (creates an unordered tuple, binding with an element already in the tuple removes it)"""
        return x1 != x2

    @staticmethod
    def bind_all(xs: list[np.ndarray]) -> np.ndarray:
        """convenience function. bind together a list of hypervectors"""
        result = np.bitwise_xor.reduce(xs)
        assert (result.shape == xs[0].shape)
        return result


    @staticmethod
    def bundle(xs):
        """bundle together xs, a list of hypervectors"""
        num_hvs = len(xs)
        counts = np.zeros((HDC.SIZE,), 'int')
        for x in xs:
            counts += x
        return np.where(counts > num_hvs / 2,
                        1,
                        np.where(counts < num_hvs / 2,
                                 0,
                                 np.random.uniform(0.0, 1.0, HDC.SIZE) < 0.5))

    @staticmethod
    def permute(x, i):
        """permute x by i, where i can be positive or negative"""
        return np.roll(x, i)

    @staticmethod
    def apply_bit_flips(x, p=0.0):
        """return a corrupted hypervector, given a per-bit bit flip probability p"""
        return np.where(np.random.uniform(0.0, 1.0, HDC.SIZE) < p,
                        ~x,
                        x)


class HDItemMem:
    def __init__(self, name=None) -> None:
        self.name = name
        self.item_mem = {str(): np.zeros(HDC.SIZE, 'bool') for _ in range(0)}
        # per-bit bit flip probabilities for the  hamming distance
        self.prob_bit_flips = 0.0

    def add(self, key, hv):
        assert (hv is not None)
        # Apply bit flip errors to hypervectors before they are stored in item memory
        # This probability is always 0.01
        hv = HDC.apply_bit_flips(hv, p=0.01)
        self.item_mem[key] = hv

    def get(self, key):
        return self.item_mem[key]

    def has(self, key):
        return key in self.item_mem

    def distance(self, query):
        """compute hamming distance between query vector and each row in item memory. Introduce bit flips if the bit flip probability is nonzero"""
        # print(f'Applying bit flips with probability {self.prob_bit_flips}')
        return {k: HDC.dist(v, HDC.apply_bit_flips(query, p=self.prob_bit_flips))
                for k, v in self.item_mem.items()}

    def all_keys(self):
        return list(self.item_mem.keys())

    def all_hvs(self):
        return list(self.item_mem.values())

    def wta(self, query):
        """winner-take-all querying"""
        distances = self.distance(query)
        key = min(distances.keys(), key=distances.get)
        dist = distances[key]
        return key, dist

    def matches(self, query, threshold=0.49):
        """threshold-based querying"""
        distances = self.distance(query)
        return {k: v for k, v in distances.items() if v < threshold}

    # a codebook is simply an item memory that always creates a random hypervector

# when a key is added.
class HDCodebook(HDItemMem):
    def add(self, key):
        super().add(key, HDC.rand_vec())
        # self.item_mem[key] = HDC.rand_vec()


def make_letter_hvs() -> HDCodebook:
    """return a codebook of letter hypervectors"""
    letter_cb = HDCodebook(name='letter_cb')
    for letter in string.ascii_lowercase:
        letter_cb.add(letter)
    return letter_cb


def make_word(letter_codebook: HDCodebook, word: str) -> np.ndarray:
    """make a word using the letter codebook"""
    letter_hvs = [
        HDC.permute(letter_codebook.get(c), i)
        for i, c in enumerate(word)
    ]
    # Don't permute
    # letter_hvs = [
    #     letter_codebook.get(c)
    #     for c in word
    # ]
    return HDC.bundle(letter_hvs)



def monte_carlo(fxn, trials: int):
    results = list(map(lambda i: fxn(), tqdm.trange(trials)))
    return results


def plot_dist_distributions(key1, dist1, key2, dist2):
    plt.hist(dist1,
             alpha=0.75,
             label=key1)

    plt.hist(dist2,
             alpha=0.75,
             label=key2)

    plt.legend(loc='upper right')
    plt.title('Distance distribution for Two Words')
    plt.show()
    plt.clf()


def study_distributions():
    def gen_codebook_and_words(w1, w2, prob_error=0.0) -> float:
        """encode words and compute distance"""
        letter_cb = make_letter_hvs()
        letter_cb.prob_bit_flips = prob_error
        hv1 = make_word(letter_cb, w1)
        hv2 = make_word(letter_cb, w2)
        word_cb = HDItemMem(name='word_cb')
        word_cb.prob_bit_flips = prob_error
        word_cb.add(w1, hv1)
        word_cb.add(w2, hv2)
        return word_cb.distance(hv2)[w1]

    trials = 1000
    d1 = monte_carlo(lambda: gen_codebook_and_words("fox", "box"), trials)
    d2 = monte_carlo(lambda: gen_codebook_and_words("fox", "car"), trials)
    plot_dist_distributions("box", d1, "car", d2)

    perr = 0.10
    d1 = monte_carlo(lambda: gen_codebook_and_words("fox", "box", prob_error=perr), trials)
    d2 = monte_carlo(lambda: gen_codebook_and_words("fox", "car", prob_error=perr), trials)
    plot_dist_distributions("box", d1, "car", d2)


if __name__ == '__main__':
    HDC.SIZE = 10000

    letter_cb = make_letter_hvs()
    hv1 = make_word(letter_cb, "fox")
    hv2 = make_word(letter_cb, "box")
    hv3 = make_word(letter_cb, "xfo")
    hv4 = make_word(letter_cb, "car")

    print(HDC.dist(hv1, hv1))
    print(HDC.dist(hv1, hv2))
    print(HDC.dist(hv1, hv3))
    print(HDC.dist(hv1, hv4))

    study_distributions()
