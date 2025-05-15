"""
скрипт, позволяющий автоматически запустить препроцессинг данных

в качестве аргументов через командную строку передаются полный путь до .zip архива с разметкой фотографий, а также
полный путь до папки, в которой будут сохранены маски для интересующих классов
"""

import argparse
import cv2
import json
import numpy as np
import zipfile

from pathlib import Path
from PIL import Image, ImageOps

# список интересующих классов для сегментации
ALL_LABELS = [
    'ВсяРана', 'Фибрин', 'Металлоконструкция', 'Зона шва',
    'Зона отека вокруг раны', 'Зона гиперемии вокруг',
    'Зона некроза', 'Зона грануляций', 'Вторичная пигментация',
    'Подкожная жир.кл. без грануляций', 'Фасция без грануляций',
    'Сухожилие', 'Гнойное отделяемое'
]


def process_points(points, scale):
    # обработка точек для корректного преобразования маски
    scaled_polygon = []
    for i in range(0, len(points), 2):
        x_orig = points[i]
        y_orig = points[i + 1]
        x_new = x_orig * scale
        y_new = y_orig * scale
        scaled_polygon.append([x_new, y_new])
    return np.array(scaled_polygon, dtype=np.int32)


def parse_manifest(path):
    # извлечение информации о фотографиях: имя, расширение, ширина и высота
    results = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line:
                continue
            obj = json.loads(line)

            if 'name' not in obj:
                continue
            results.append((obj['name'] + obj['extension'], obj['width'], obj['height']))
    return results


def create_masks_and_resized_image(image_path, orig_width, orig_height, shapes, output_image_path,
                                   output_masks_path, base_name):
    # изменение размеров фотографии так, чтобы большая сторона стала равна 1024, а пропорции сохранились
    target_size = 1024
    long_side = max(orig_width, orig_height)
    scale = target_size / long_side
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))

    image = ImageOps.exif_transpose(Image.open(image_path))
    resized_image = image.resize((new_width, new_height))

    # сохранение обработанной фотографии
    final_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    final_image.paste(resized_image, (0, 0))
    final_image.save(output_image_path)

    # создание масок для каждого из рассматриваемых классов
    masks = {label: np.zeros((target_size, target_size), dtype=np.uint8) for label in ALL_LABELS}

    for shape in shapes:
        if shape['type'] == 'polygon':
            flat_points = shape['points']
            # преобразование точек, описывающих контур маски
            polygon_points = process_points(flat_points, scale)
            label_name = shape['label']

            if label_name in masks:
                # сохранение обработанной маски
                cv2.fillPoly(masks[label_name], [polygon_points], color=255)
                success, encoded_mask = cv2.imencode('.png', masks[label_name])
                if success:
                    current_mask_path = output_masks_path / label_name
                    current_mask_path.mkdir(parents=True, exist_ok=True)
                    current_mask_path = current_mask_path / f'{base_name}.png'
                    with open(current_mask_path, 'wb') as f:
                        f.write(encoded_mask)


def process_folder(root_folder, output_folder):
    # создание папок для сохранения исходных фотографий и соответствующих им масок для всех классов
    images_directory = (root_folder / f'{output_folder}/images')
    masks_directory = (root_folder / f'{output_folder}/masks')
    images_directory.mkdir(parents=True, exist_ok=True)
    masks_directory.mkdir(parents=True, exist_ok=True)

    # обработка каждой из фотографий
    for task in root_folder.glob('task_*'):
        annotations_path = task / 'annotations.json'
        data_directory = task / 'data'
        manifest_path = data_directory / 'manifest.jsonl'

        with open(annotations_path, 'r', encoding='utf-8') as file:
            annotations = json.load(file)

        # сохранение многоугольников из описанной попиксельной разметки
        shapes_by_frame = {}
        for entry in annotations:
            shapes = entry.get('shapes', [])
            for shape in shapes:
                frame_index = shape['frame']
                if frame_index not in shapes_by_frame:
                    shapes_by_frame[frame_index] = []
                shapes_by_frame[frame_index].append(shape)

        # извлечение информации о фотографиях из файла manifest.jsonl
        manifest = parse_manifest(manifest_path)

        for frame_index, (filename, width, height) in enumerate(manifest):
            image_path = data_directory / filename

            base_name = f'{task.name}_{filename.split(".")[0]}'
            shapes = shapes_by_frame.get(frame_index, [])

            # сохранение оригинальных фотографий и масок в нужном формате
            create_masks_and_resized_image(
                image_path=image_path,
                orig_width=width,
                orig_height=height,
                shapes=shapes,
                output_image_path=images_directory / f'{base_name}.png',
                output_masks_path=masks_directory,
                base_name=base_name
            )


def main():
    # обработка аргументов из командной строки
    parser = argparse.ArgumentParser(description="Images preprocessing")
    parser.add_argument('path_to_zip_file', help='Path to .zip file')
    parser.add_argument('output_folder', help='Path to the folder for saving masks')
    args = parser.parse_args()

    # извлечение всех данных из имеющегося .zip файла
    extracted_directory = 'tasks'
    with zipfile.ZipFile(args.path_to_zip_file) as file:
        file.extractall(extracted_directory)

    # обработка разархивированных фотографий
    process_folder(Path(extracted_directory), Path(args.output_folder))


if __name__ == '__main__':
    main()
