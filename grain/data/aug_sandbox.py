from grain.configs import *
import cv2
import imgaug as ia
from imgaug import augmenters as iaa



seq_default = iaa.Sequential([
    iaa.Crop(percent=(0, 0.2)),
    iaa.Fliplr(0.5),
    iaa.ContrastNormalization((0.8, 1.3)),
    iaa.Add((-20, 20), per_channel=0.8),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04 * 255), per_channel=0.5)
])


if __name__ == '__main__':
    PIC_LIST = os.listdir(PICTURES_FOLDER)
    MASK_LIST = os.listdir(MASK_FOLDER)

    PIC_LIST.sort()
    MASK_LIST.sort()

    img = cv2.imread(os.path.join(PICTURES_FOLDER, PIC_LIST[0]), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(MASK_FOLDER, MASK_LIST[0]), cv2.IMREAD_GRAYSCALE)

    while True:
        seq_det = seq_default.to_deterministic()

        aug_img = seq_det.augment_image(img)
        aug_mask = seq_det.augment_image(mask)
        aug_mask = cv2.threshold(aug_mask, 127, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.imshow('aug_img', aug_img)
        cv2.imshow('aug_mask', aug_mask)

        w = 4 * WIDTH // 3
        h = 3 * HEIGHT // 2

        cv2.moveWindow('img', 0, 0)
        cv2.moveWindow('mask', w, 0)
        cv2.moveWindow('aug_img', 0, h)
        cv2.moveWindow('aug_mask', w, h)

        if cv2.waitKey(0) == ord('q'):
            break
