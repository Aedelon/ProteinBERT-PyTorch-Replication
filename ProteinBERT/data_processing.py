#!/usr/bin/env python
# *** coding: utf-8 ***

"""data_processing.py: Contains the different classes and functions which will be used to preprocess the data.

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT

   * Paper: "ProteinBERT: A universal deep-learning model of protein sequence and function. "
   * Paper's authors: Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. and Linial, M. .
   * Paper DOI: https://doi.org/10.1093/bioinformatics/btac020
"""

# IMPORTS -------------------------------------------------
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torchtext

from pathlib import Path
from torchtext import transforms
from torch.utils import data


# CLASSES -------------------------------------------------
# CUSTOM TRANSFORMS ---
class SimpleCharacterTokenizer(torch.nn.Module):
    """Numericalize sentence with a simple character vocabulary.

    Args:
        vocab (torchtext.vocab.Vocab): Vocabulary of characters.
        add_sos_eos (bool): If True, add <sos> token at the beginning of the sample and <eos> at the end of the sample.
        sos_index (int): If add_sos_eos is True then this attribute shows the index of <sos> token in the vocabulary.
        eos_index (int): If add_sos_eos is True then this attribute shows the index of <eos> token in the vocabulary.
    """

    def __init__(self,
                 vocab: torchtext.vocab.Vocab,
                 add_sos_eos: bool = True,
                 sos_index: int = 1,
                 eos_index: int = 2):
        assert isinstance(vocab, torchtext.vocab.Vocab)
        assert isinstance(add_sos_eos, bool)
        super().__init__()

        if add_sos_eos:
            self.tokenizer = transforms.Sequential(
                transforms.VocabTransform(vocab=vocab),
                transforms.AddToken(sos_index, begin=True),
                transforms.AddToken(eos_index, begin=False)
            )
        else:
            self.tokenizer = transforms.Sequential(
                transforms.VocabTransform(vocab=vocab)
            )

    def __call__(self, sample):
        return self.tokenizer(list(sample))


class SentenceRandomCrop(torch.nn.Module):
    """Take a chunk of the sentence and return it. If the sample has an array length below max_length then it return
    the sample.

    Args:
        max_length (int): The length of the desired chunk.
    """

    def __init__(self, max_length: int):
        assert isinstance(max_length, int)
        super().__init__()

        self.max_length = max_length

    def __call__(self, sample):
        if len(sample) <= self.max_length:
            return sample

        start_index = torch.randint(0, (len(sample) - self.max_length), (1,))[0]
        return sample[start_index:(start_index + self.max_length)]


class SimpleTokenRandomizer(torch.nn.Module):
    """Randomize tokens from the sentence with a random token.

    """

    def __init__(self, vocab: torchtext.vocab.Vocab, p: float = .05):
        assert isinstance(vocab, torchtext.vocab.Vocab)
        assert isinstance(p, float)
        super().__init__()

        self.vocab = vocab
        self.p = p
        self.exclude_tokens = (0, 1, 2)

    def __call__(self, sample):
        mask = torch.multinomial(torch.tensor([1 - self.p, self.p], dtype=torch.float), num_samples=sample.shape[0], replacement=True)
        for token_id in self.exclude_tokens:
            mask = mask & (sample != token_id)
        random_seq_tokens = torch.randint(3, len(self.vocab), mask.shape)
        return torch.where(mask.bool(), random_seq_tokens, sample)


class AnnotationMasking(torch.nn.Module):
    """Corrupt GO annotations by randomly removing existing annotations with probability (called positive probability),
    and adding random false annotations with probability (called negative probability) for each annotation not
    associated with the protein.

    Args:

    """

    def __init__(self, positive_p: float = 0.25, negative_p: float = 0.0001):
        assert isinstance(positive_p, float)
        assert isinstance(negative_p, float)
        super().__init__()

        self.positive_p = positive_p
        self.negative_p = negative_p

    def __call__(self, sample):
        sample = torch.tensor(sample)
        noised_annotation = torch.zeros(sample.shape)
        if torch.rand(1)[0] > 0.5:
            noisy_positive_annotations = torch.multinomial(
                torch.tensor([self.positive_p, 1 - self.positive_p], dtype=torch.float),
                num_samples=sample.shape[0],
                replacement=True
            )
            noisy_negative_annotations = torch.multinomial(
                torch.tensor([1 - self.negative_p, self.negative_p], dtype=torch.float),
                num_samples=sample.shape[0],
                replacement=True
            )
            noised_annotation = sample + noisy_negative_annotations.type(sample.dtype)
            noised_annotation = noised_annotation * noisy_positive_annotations.type(sample.dtype)

        return noised_annotation


# CUSTOM DATASET ---
class UniRefGO_PretrainingDataset(data.Dataset):
    def __init__(self, df, seq_max_length=128):
        self.df = df
        self.vocab = create_amino_acid_vocab()
        self.sequences_transform = torchtext.transforms.Sequential(
            SimpleCharacterTokenizer(vocab=self.vocab),
            SentenceRandomCrop(max_length=seq_max_length),
            transforms.ToTensor(padding_value=seq_max_length),
        )
        self.pad_transform = torchtext.transforms.PadTransform(max_length=seq_max_length, pad_value=self.vocab["<pad>"])
        self.token_randomizer = SimpleTokenRandomizer(vocab=self.vocab, p=.05)
        self.annotations_masking = AnnotationMasking(positive_p=0.25, negative_p=0.0001)

    def __getitem__(self, index):
        # get data
        seq = self.df.iloc[index, 0]
        seq = self.sequences_transform(seq)

        masked_seq = self.token_randomizer(seq)
        masked_seq = self.pad_transform(masked_seq)

        seq = self.pad_transform(seq)

        # get label
        ann = self.df.iloc[index, 1]

        masked_ann = self.annotations_masking(ann)
        ann = torch.tensor(ann)

        seq_weights = (seq.numpy() != self.vocab["<pad>"]).astype(float)
        annotation_weights = ann.numpy().any(axis=-1).astype(float).repeat(ann.shape)

        return {"local": masked_seq, "global": masked_ann.float()}, \
               {"local": seq, "global": ann.float()}, \
               {"local": seq_weights, "global": annotation_weights}

    def __len__(self):
        return len(self.df.index)


class UniRefGO_HDF5PretrainingDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        seq_max_length: Maximum length of for the protein sequence (default=128).

    Inspiration source: Branislav Holländer - https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    """

    def __init__(self,
                 file_path,
                 recursive,
                 load_data,
                 data_cache_size=3,
                 seq_max_length=128):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.vocab = create_amino_acid_vocab()
        self.sequences_transform = torchtext.transforms.Sequential(
            SimpleCharacterTokenizer(vocab=self.vocab),
            SentenceRandomCrop(max_length=seq_max_length),
            transforms.ToTensor(padding_value=seq_max_length),
        )
        self.pad_transform = torchtext.transforms.PadTransform(max_length=seq_max_length, pad_value=self.vocab["<pad>"])
        self.token_randomizer = SimpleTokenRandomizer(vocab=self.vocab, p=.05)
        self.annotations_masking = AnnotationMasking(positive_p=0.25, negative_p=0.0001)

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        seq = self.get_data("seqs", index)
        seq = self.sequences_transform(seq)

        masked_seq = self.token_randomizer(seq)
        masked_seq = self.pad_transform(masked_seq)

        # get label
        ann = self.get_data("annotation_masks", index)

        masked_ann = self.annotations_masking(ann)
        ann = torch.tensor(ann)

        seq_weights = (seq != self.vocab["<pad>"]).astype(float)
        annotation_weights = ann.any(axis=-1).astype(float)

        return {"local": masked_seq, "global": masked_ann}, {"local": seq, "global": ann}, \
               {"local": seq_weights, "global": annotation_weights}

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


# FUNCTIONS -----------------------------------------------
def create_amino_acid_vocab():
    all_amino_acids = "ACDEFGHIKLMNPQRSTUVWXY"
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    aa_vocab = torchtext.vocab.vocab(
        OrderedDict((token, 1) for token in list(all_amino_acids)),
        specials=special_tokens
    )
    aa_vocab.set_default_index(aa_vocab["<unk>"])

    assert aa_vocab["out of vocab"] is aa_vocab["<unk>"]
    return aa_vocab
