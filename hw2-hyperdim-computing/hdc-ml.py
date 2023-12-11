from torchvision.datasets import MNIST
from torch.utils.data import Subset
import tqdm
import torch.utils
from hdc import *
import itertools
import PIL
import math
import numpy as np

MAX = 28


class MNISTClassifier:
    def __init__(self):
        self.classifier = HDItemMem()
        self.classifier.prob_bit_flips = 0.29
        # initialize other stuff here
        self.c_codebook = HDCodebook()
        # self.y_codebook = HDCodebook()
        self.v_codebook = HDCodebook()

    def encode_coord(self, i, j) -> np.ndarray:
        """encode a coordinate in the image as a hypervector"""
        if not self.c_codebook.has(i):
            self.c_codebook.add(i)
        if not self.c_codebook.has(j):
            self.c_codebook.add(j)
        return HDC.bind(self.c_codebook.get(i), HDC.permute(self.c_codebook.get(j), 1))

    def encode_pixel(self, i, j, v):
        """encode a pixel in the image as a hypervector"""
        coord_hv = self.encode_coord(i, j)
        if not self.v_codebook.has(v):
            self.v_codebook.add(v)
        v_hv = self.v_codebook.get(v)
        pixel_hv = HDC.bind(coord_hv, v_hv)
        return pixel_hv

    def encode_image(self, image) -> np.ndarray:
        """return hypervector encoding of image"""
        pixel_hvs = []
        for i, j in itertools.product(range(MAX), range(MAX)):
            v = image.getpixel((i, j))
            # print("TODO: do something with encoded pixel value")
            pixel_hv = self.encode_pixel(i, j, v)
            pixel_hvs.append(pixel_hv)
        return HDC.bundle(pixel_hvs)

    def decode_pixel(self, image_hypervec, i, j):
        """retrieve the value of the pixel at coordinate i,j in the image hypervector"""
        coord_hv = self.encode_coord(i, j)
        pixel_hv = HDC.bind(image_hypervec, coord_hv)
        return self.v_codebook.wta(pixel_hv)[0]

    def decode_image(self, image_hypervec):
        im = PIL.Image.new(mode="1", size=(MAX, MAX))
        for i, j in list(itertools.product(range(MAX), range(MAX))):
            v = self.decode_pixel(image_hypervec, i, j)
            im.putpixel((i, j), v)
        return im

    def train(self, train_data):
        from collections import defaultdict
        label_hvs = defaultdict(list)
        for image, label in tqdm.tqdm(train_data):
            # do something with the image,label pair from the dataset
            image_hv = self.encode_image(image)
            label_hvs[label].append(image_hv)
        for k, hvs in label_hvs.items():
            self.classifier.add(k, HDC.bundle(hvs))

    def classify(self, image):
        """classify an image using your classifier and return the label and distance"""
        image_hv = self.encode_image(image)
        label, dist = self.classifier.wta(image_hv)
        return label, dist

    def build_gen_model(self, train_data):
        """build generative model"""
        from collections import defaultdict
        self.gen_model = {}
        label_hvs = defaultdict(list)
        for image, label in tqdm.tqdm(list(train_data)):
            label_hvs[label].append(self.encode_image(image))
        self.gen_model = {k: np.mean(hvs, axis=0) for k, hvs in label_hvs.items()}

    def generate(self, cat, trials=10):
        """generate random image with label <cat> using generative model. Average over <trials> trials."""
        gen_hv = self.gen_model[cat]
        generate_hvs = [np.random.uniform(size=gen_hv.shape) < gen_hv for _ in range(trials)]
        return self.decode_image(HDC.bundle(generate_hvs))


def initialize(N=1000):
    alldata = MNIST(root='data', train=True, download=True)
    dataset = list(map(lambda datum: (datum[0].convert("1"), datum[1]),
                       Subset(alldata, range(N))))

    train_data, test_data = torch.utils.data.random_split(dataset, [0.6, 0.4])
    HDC.SIZE = 10_000
    classifier = MNISTClassifier()
    return train_data, test_data, classifier


# @torch.compile
def test_encoding():
    train_data, test_data, classifier = initialize()
    image0, _ = train_data[0]
    hv_image0 = classifier.encode_image(image0)
    result = classifier.decode_image(hv_image0)
    image0.save("sample0.png")
    result.save("sample0_rec.png")


def test_classifier():
    train_data, test_data, classifier = initialize(2000)

    print("======= training classifier =====")
    classifier.train(train_data)

    print("======= testing classifier =====")
    correct, count = 0, 0
    for image, category in (pbar := tqdm.tqdm(test_data)):
        cat, dist = classifier.classify(image)
        if cat == category:
            correct += 1
        count += 1

        pbar.set_description("accuracy=%f" % (float(correct) / count))

    print("ACCURACY: %f" % (float(correct) / count))


def test_generative_model():
    train_data, test_data, classifier = initialize(1000)
    print("======= building generative model =====")
    classifier.build_gen_model(train_data)

    print("======= generate images =====")
    for i in range(10):
        cat = random.randint(0, 9)
        img = classifier.generate(cat)
        print("generated image for class %d" % cat)
        img.save(f"generated_cat{cat}_idx{i}.png")
        # input("press any key to generate new image..")


if __name__ == '__main__':
    test_encoding()
    test_classifier()
    # test_generative_model()
