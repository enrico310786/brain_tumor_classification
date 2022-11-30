import cv2
import albumentations as A
import random
img_size = 224

PATH_IMAGE = "../resources/image_test/draghi.jpeg"


transform = A.Compose([
    A.CLAHE(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ToGray(p=0.5),
    A.RandomGamma(p=0.5),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
    A.Resize(img_size, img_size, p=1.)
])

image = cv2.imread(PATH_IMAGE)
img = cv2.resize(image, (img_size, img_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



for i in range(10):
    augmented_image = transform(image=image)['image']
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("resources/image_test/draghi_" + str(i+1) + ".png", augmented_image)