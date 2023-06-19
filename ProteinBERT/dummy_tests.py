# IMPORTS -------------------------------------------------
import logging
import random
import pandas as pd
import torch
import torchtext
import data_processing

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCELoss
from torchinfo import summary
from modules import ProteinBERT
from utils import pretrain

# CONSTANTS -----------------------------------------------
ALL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTUVWXY"
NB_ANNOTATIONS = 8943
NUM_WORKERS = 0
BATCH_SIZE = 32


# FUNCTIONS -----------------------------------------------
def create_random_samples(nb_samples, seed=7777):
    samples = []
    random.seed(seed)
    for i in range(nb_samples):
        sample_length = random.randint(0, 250)
        sample_seq = ''
        for j in range(sample_length):
            sample_seq += ALL_AMINO_ACIDS[random.randint(0, len(ALL_AMINO_ACIDS) - 1)]

        sample_ann = []
        for j in range(NB_ANNOTATIONS):
            sample_ann.append(0 if random.random() * 1000 > 5 else 1)

        samples.append((sample_seq, sample_ann))

    return samples


def test_sequence_transform(samples, vocab, seq_max_length):
    sequences_transform = torchtext.transforms.Sequential(
        data_processing.SimpleCharacterTokenizer(vocab=vocab),
        data_processing.SentenceRandomCrop(max_length=seq_max_length),
        torchtext.transforms.ToTensor(padding_value=seq_max_length),
    )

    new_samples = []
    for sample in samples:
        new_samples.append((sequences_transform(sample[0]), sample[1]))

    return new_samples


def test_sequence_token_randomizer(samples, vocab):
    token_randomizer = data_processing.SimpleTokenRandomizer(vocab=vocab, p=.05)

    new_samples = []
    for sample in samples:
        new_samples.append((token_randomizer(sample[0]), sample[1]))

    return new_samples


def test_masking_annotation(samples):
    token_randomizer = data_processing.AnnotationMasking()

    new_samples = []
    for sample in samples:
        new_samples.append((sample[0], token_randomizer(sample[1])))

    return new_samples


def test_data_processing():
    samples = create_random_samples(5)
    vocab = data_processing.create_amino_acid_vocab()

    print("VOCAB:")
    print(vocab.get_itos())
    seq_transformation_samples = test_sequence_transform(samples=samples, vocab=vocab, seq_max_length=128)
    print("TRANSFORMED SEQUENCES:")
    for sample in seq_transformation_samples:
        print(sample[0], "\n")
    print("RANDOMIZED TOKEN SEQUENCES:")
    randomized_token_seq_samples = test_sequence_token_randomizer(samples=seq_transformation_samples, vocab=vocab)
    for sample in randomized_token_seq_samples:
        print(sample[0], "\n")

    print("MASKED ANNOTATIONS:")
    final_samples = test_masking_annotation(samples)
    for sample in final_samples:
        print(sample[1], "\n")


def main():
    samples = create_random_samples(100, 1)
    df = pd.DataFrame(samples)

    print(df)

    dataset = data_processing.UniRefGO_PretrainingDataset(df, seq_max_length=256)
    train_dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    )

    model = ProteinBERT(
        sequences_length=256,
        num_annotations=NB_ANNOTATIONS,
        local_dim=128,
        global_dim=512,
        key_dim=64,
        num_heads=4,
        num_blocks=6
    )

    print(f"""\n {summary(model=model,
                          col_names=["num_params", "trainable"],
                          col_width=20,
                          row_settings=["var_names"],
                          verbose=0)}
          """)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=2e-04
    )

    local_loss_fn = CrossEntropyLoss(reduction='none')
    global_loss_fn = BCELoss(reduction='none')

    pretrain(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        local_loss_fn=local_loss_fn,
        global_loss_fn=global_loss_fn,
        max_batch_iterations=250,
        save_path='.'
    )


if __name__ == '__main__':
    logging.basicConfig(
        # filename=f"{save_path / 'pretraining.log'}",
        # filemode='w',
        format="%(asctime)s [%(levelname)s]: %(message)s",
        level=logging.INFO
    )

    logging.info(f"Test pretraining initialization...")
    main()
