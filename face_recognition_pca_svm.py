import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------------
# Load Dataset
# -----------------------------------

print("Loading AT&T Dataset...")

dataset_path = r"C:\ad lab\lab10\ATnT"

X = []
y = []
img_shape = None

for label_folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, label_folder)

    if os.path.isdir(folder_path):

        label = int(label_folder.replace("s",""))

        for image_file in os.listdir(folder_path):

            if image_file.endswith(".pgm"):

                img_path = os.path.join(folder_path, image_file)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:

                    img_shape = img.shape
                    X.append(img.flatten())
                    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total Images:", X.shape[0])
print("Image Size:", img_shape)
print("Feature Length:", X.shape[1])


# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)


# -----------------------------------
# PCA using EVD
# -----------------------------------

def pca_evd(X_train):

    mean_face = np.mean(X_train, axis=0)

    X_centered = X_train - mean_face

    small_cov = np.dot(X_centered, X_centered.T) / (X_train.shape[0]-1)

    eigenvalues, small_eigenvectors = np.linalg.eigh(small_cov)

    idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[idx]
    small_eigenvectors = small_eigenvectors[:,idx]

    eigenvectors = np.dot(X_centered.T, small_eigenvectors)

    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    return mean_face, eigenvalues, eigenvectors


print("\nRunning PCA using EVD...")

start = time.time()

mean_face_evd, eig_vals_evd, eig_vecs_evd = pca_evd(X_train)

evd_time = time.time() - start

print("PCA Completed in", round(evd_time,4), "seconds")


# -----------------------------------
# Scree Plot (Eigenvalues)
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(eig_vals_evd[:50], marker='o')

plt.title("Scree Plot — Eigenvalues")

plt.xlabel("Component Number")

plt.ylabel("Eigenvalue")

plt.grid(True)

plt.show()


# -----------------------------------
# Cumulative Variance Plot
# -----------------------------------

explained_var = eig_vals_evd / np.sum(eig_vals_evd)

cum_var = np.cumsum(explained_var)

plt.figure(figsize=(8,5))

plt.plot(cum_var[:100], label="EVD")

plt.axhline(y=0.90, linestyle="--", label="90% variance")

plt.axhline(y=0.95, linestyle=":", label="95% variance")

plt.xlabel("Number of Components")

plt.ylabel("Cumulative Variance Explained")

plt.title("Cumulative Variance")

plt.legend()

plt.grid(True)

plt.show()


# -----------------------------------
# Mean Face
# -----------------------------------

plt.figure(figsize=(4,4))

plt.imshow(mean_face_evd.reshape(img_shape), cmap='gray')

plt.title("Mean Face")

plt.axis('off')

plt.show()


# -----------------------------------
# Eigenfaces
# -----------------------------------

plt.figure(figsize=(10,4))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(eig_vecs_evd[:,i].reshape(img_shape), cmap='seismic')

    plt.title("EF "+str(i+1))

    plt.axis('off')

plt.suptitle("Mean Face & Top Eigenfaces (EVD)")

plt.show()


# -----------------------------------
# Image Reconstruction
# -----------------------------------

def reconstruct(image, mean, eig_vecs, k):

    img_centered = image - mean

    V = eig_vecs[:,:k]

    projection = np.dot(img_centered, V)

    reconstruction = np.dot(projection, V.T) + mean

    return reconstruction


test_img = X_test[0]

components = [10,50,100,200]

plt.figure(figsize=(15,4))

plt.subplot(1,len(components)+1,1)

plt.imshow(test_img.reshape(img_shape), cmap='gray')

plt.title("Original")

plt.axis('off')

for i,k in enumerate(components):

    rec = reconstruct(test_img, mean_face_evd, eig_vecs_evd, k)

    plt.subplot(1,len(components)+1,i+2)

    plt.imshow(rec.reshape(img_shape), cmap='gray')

    plt.title(str(k)+" comps")

    plt.axis('off')

plt.show()


# -----------------------------------
# SVM Face Recognition
# -----------------------------------

print("\nTraining SVM Classifier...")

components_list = [5,10,20,50,100,150]

accuracies = []

for k in components_list:

    V = eig_vecs_evd[:,:k]

    X_train_pca = np.dot(X_train - mean_face_evd, V)

    X_test_pca = np.dot(X_test - mean_face_evd, V)

    svm = SVC(kernel='linear')

    svm.fit(X_train_pca, y_train)

    y_pred = svm.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

    print("Components:",k,"Accuracy:",round(acc*100,2),"%")

plt.figure(figsize=(8,5))

plt.plot(components_list, accuracies, marker='o')

plt.xlabel("Principal Components")

plt.ylabel("Recognition Accuracy")

plt.title("SVM Accuracy vs PCA Components")

plt.grid(True)

plt.show()


# -----------------------------------
# PCA using SVD
# -----------------------------------

print("\nRunning PCA using SVD...")

def pca_svd(X_train):

    mean_face = np.mean(X_train, axis=0)

    X_centered = X_train - mean_face

    U,S,Vt = np.linalg.svd(X_centered, full_matrices=False)

    return mean_face, Vt.T


start = time.time()

mean_face_svd, eig_vecs_svd = pca_svd(X_train)

svd_time = time.time() - start


# -----------------------------------
# Time Comparison
# -----------------------------------

print("\nComputation Time Comparison")

print("EVD Time:", round(evd_time,4),"seconds")

print("SVD Time:", round(svd_time,4),"seconds")

if svd_time < evd_time:
    print("SVD is faster")
else:
    print("EVD is faster")