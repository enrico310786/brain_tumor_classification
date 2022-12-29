import cv2
import albumentations as A

PATH_IMAGE = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/train/glioma_tumor/gg (135).jpg"

transform = A.Compose([
    A.HueSaturationValue(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
    A.Transpose(p=0.5),
])

image = cv2.imread(PATH_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
name_file_no_ext = PATH_IMAGE.split("/")[-1].split(".")[0]
file_ext = PATH_IMAGE.split("/")[-1].split(".")[-1]

for i in range(10):
    augmented_image = transform(image=image)['image']
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("resources/image_test/" + name_file_no_ext + "_" + str(i+1) + "." + file_ext, augmented_image)