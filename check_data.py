import cv2

path_img = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/train/glioma_tumor/gg (2).jpg"
img = cv2.imread(path_img)

print("img.shape: ", img.shape)
print("img.dtype: ", img.dtype)

cv2.imshow("image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
