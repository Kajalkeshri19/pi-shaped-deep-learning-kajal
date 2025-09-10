#  Fashion-MNIST CNN Classifier

This project builds and trains a **Convolutional Neural Network (CNN)** on the **Fashion-MNIST dataset** to classify images of clothing into 10 categories (e.g., t-shirt, trouser, sneaker, bag).  
The model is trained using TensorFlow/Keras and evaluated using accuracy, confusion matrix, and classification metrics.

---

##  Dataset
- **Fashion-MNIST**: 60,000 training images + 10,000 test images  
- Image size: **28x28 grayscale**  
- Classes: 10 fashion categories  

---

##  Project Steps
1. Load the **Fashion-MNIST dataset**  
2. Preprocess the data:  
   - Normalize pixel values to `[0,1]`  
   - Reshape inputs to `(28, 28, 1)` for CNN layers  
   - One-hot encode labels  
3. Build a simple CNN with:  
   - Convolutional + MaxPooling layers  
   - Dense layers with **softmax output** for 10 classes  
4. Compile the model with Adam optimizer and categorical crossentropy  
5. Train the model on training set and validate on test set  
6. Evaluate the model using:  
   - Accuracy  
   - Confusion matrix  
   - Classification metrics  

---

##  Results
- Typical accuracy: **~89–91%** on test set  
- Confusion matrix shows per-class performance  
- Training curves used to monitor overfitting  

---

##  Core Concept Questions

**1. What advantages do CNNs have over traditional fully connected neural networks for image data?**  
- CNNs exploit the **spatial structure** of images.  
- Use **convolutional filters** to detect local patterns (edges, textures, shapes).  
- Require fewer parameters → more efficient, less prone to overfitting.  

---

**2. What is the role of convolutional filters/kernels in a CNN?**  
- Filters slide over the image to **extract features** like edges, corners, textures.  
- Each filter learns to detect a specific pattern.  
- Deeper layers capture more complex features.  

---

**3. Why do we use pooling layers, and what is the difference between MaxPooling and AveragePooling?**  
- Pooling reduces **spatial dimensions** of feature maps.  
- **MaxPooling**: picks the strongest feature in a region.  
- **AveragePooling**: averages values in a region (smoother representation).  

---

**4. Why is normalization of image pixels important before training?**  
- Ensures **faster convergence** and stable training.  
- Prevents large pixel values from causing unstable gradients.  

---

**5. How does the softmax activation function work in multi-class classification?**  
- Converts logits into **probabilities**.  
- Formula:  
  \[
  P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
  \]  
- Outputs sum to 1 → interpretable as class probabilities.  

---

**6. What strategies can help prevent overfitting in CNNs?**  
- **Dropout** (randomly turning off neurons).  
- **Data augmentation** (rotation, flips, zoom).  
- **Early stopping** (stop training if no improvement).  
- **Regularization (L2 weight decay)**.  

---

**7. What does the confusion matrix tell you about model performance?**  
- Shows how many predictions are correct/incorrect per class.  
- Reveals **class-specific errors** (e.g., sneakers misclassified as sandals).  

---

**8. If you wanted to improve the CNN, what architectural or data changes would you try?**  
- Add more **convolutional layers** for deeper features.  
- Use **Batch Normalization**.  
- Try **different optimizers** (AdamW, RMSprop).  
- Apply **data augmentation**.  
- Use **transfer learning** (e.g., ResNet, VGG).  

---

