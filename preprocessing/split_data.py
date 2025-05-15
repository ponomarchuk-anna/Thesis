"""
скрипт, позволяющий случайным образом разделить данные на три выборки (train, val, test)

в качестве аргументов через командную строку передаются полный путь до папки с фотографиями и разметкой, а также
полный путь до папки, в которой будут созданы нужные выборки
"""

import argparse
import random
import shutil

from pathlib import Path


def split_dataset(path_to_photos, output_folder, ratios):
    random.seed(28)

    # сохранение имён всех фотографий
    image_directory = path_to_photos / 'images'
    images = [image.name for image in image_directory.glob('*.png')]

    # сохранение названий всех выделенных классов
    masks_directory = path_to_photos / 'masks'
    classes = [mask.name for mask in masks_directory.iterdir() if mask.is_dir()]

    # сохранение путей до соответствующих масок для каждой из фотографий
    photos = []
    for image_name in images:
        image_path = image_directory / image_name
        mask_paths = {}
        for current_class in classes:
            mask_path = masks_directory / current_class / image_name
            if mask_path.exists():
                mask_paths[current_class] = mask_path
        # сохранение фотографии происходит только в случае наличия для неё разметки
        if mask_paths:
            photos.append((image_path, mask_paths))

    # получение выборок train/val/test требуемого объёма
    random.shuffle(photos)
    n = len(photos)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_photos = photos[:n_train]
    val_photos = photos[n_train:n_train+n_val]
    test_photos = photos[n_train+n_val:]

    splits = {
        'train': train_photos,
        'val':   val_photos,
        'test':  test_photos
    }

    # сохранение фотографий и масок в папке, соответствующей выборке, к которой они были отнесены
    for split_name, subset in splits.items():
        image_out = output_folder / split_name / 'images'
        masks_out = output_folder / split_name / 'masks'

        image_out.mkdir(parents=True, exist_ok=True)
        for current_class in classes:
            (masks_out / current_class).mkdir(parents=True, exist_ok=True)

        for image_path, mask_paths in subset:
            shutil.copy2(image_path, image_out / image_path.name)
            for current_class, current_mask_path in mask_paths.items():
                shutil.copy2(current_mask_path, masks_out / current_class / current_mask_path.name)


def main():
    # обработка аргументов из командной строки
    parser = argparse.ArgumentParser(description="Splitting data")
    parser.add_argument('path_to_photos', help='Path to photos folder')
    parser.add_argument('output_folder', help='Path to the folder for saving data')
    args = parser.parse_args()

    # случайное разделение фотографий на три выборки
    split_dataset(
        path_to_photos=Path(args.path_to_photos),
        output_folder=Path(args.output_folder),
        ratios=(0.75, 0.10, 0.15)
    )


if __name__ == '__main__':
    main()
