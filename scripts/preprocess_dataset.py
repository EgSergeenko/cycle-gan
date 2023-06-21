import os
import random
import shutil

from sklearn.model_selection import train_test_split


def preprocess_dataset(
    content_dir: str,
    paintings_dir: str,
    output_dir: str,
    n_samples: int,
    random_seed: int,
) -> None:
    random.seed(random_seed)
    preprocess_dataset_part(
        'content',
        content_dir,
        output_dir,
        n_samples // 2,
        random_seed,
    )
    preprocess_dataset_part(
        'paintings',
        paintings_dir,
        output_dir,
        n_samples // 2,
        random_seed,
    )


def preprocess_dataset_part(
    part_name: str,
    root_dir: str,
    output_dir: str,
    n_samples: int,
    random_seed: int,
) -> None:
    filenames = os.listdir(root_dir)
    filepaths = [os.path.join(root_dir, filename) for filename in filenames]
    filepaths = random.sample(filepaths, n_samples)
    train_filepaths, test_filepaths = train_test_split(
        filepaths, test_size=0.1, random_state=random_seed, shuffle=True,
    )

    copy_files(
        os.path.join(output_dir, '{0}_train'.format(part_name), '0'),
        train_filepaths,
    )

    copy_files(
        os.path.join(output_dir, '{0}_test'.format(part_name), '0'),
        test_filepaths,
    )


def copy_files(dst_dir: str, filepaths: list[str]) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for idx, filepath in enumerate(filepaths):
        src_filepath = filepath
        dst_filepath = os.path.join(
            dst_dir, '{0}.jpg'.format(str(idx).zfill(4)),
        )
        shutil.copy(src_filepath, dst_filepath)


if __name__ == '__main__':
    preprocess_dataset(
        content_dir='../data/raw/content',
        paintings_dir='../data/raw/paintings',
        output_dir='../data/processed',
        n_samples=1000,
        random_seed=42,
    )
