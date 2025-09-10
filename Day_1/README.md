### 1. What is the role of feature scaling/normalization in training neural networks?
Feature scaling ensures that all input features are on a similar scale.  
Without scaling, features with larger numerical ranges could dominate the learning process, leading to unstable gradients and poor convergence.  
**Scaling improves training speed and stability.**

---

### 2. Why do we split data into training and testing sets?
- **Training set**: Used to teach the model patterns.  
- **Test set**: Used to evaluate how well the model generalizes to unseen data.  
This prevents overfitting and ensures fair evaluation.

---

### 3. What is the purpose of activation functions like ReLU or Sigmoid?
- **ReLU (Rectified Linear Unit):** Introduces non-linearity, prevents vanishing gradients, and allows networks to learn complex patterns.  
- **Sigmoid:** Maps output to [0,1], making it useful for **binary classification** probability predictions.  

---

### 4. Why is binary cross-entropy commonly used as a loss function for classification?
Binary cross-entropy measures the difference between predicted probabilities and actual labels.  
It penalizes confident but wrong predictions more heavily, making it ideal for **binary classification tasks**.

---

### 5. How does the optimizer (e.g., Adam) affect training compared to plain gradient descent?
- **Plain Gradient Descent:** Uses a fixed learning rate for all parameters.  
- **Adam Optimizer:** Adapts learning rates for each parameter based on momentum and past gradients.  
Result → Faster convergence, better performance on complex datasets.

---

### 6. What does the confusion matrix tell you beyond just accuracy?
The confusion matrix shows **true positives, true negatives, false positives, and false negatives**, giving deeper insight into model errors.  
It helps identify if the model favors one class (class imbalance issues) even if accuracy looks high.

---

### 7. How can increasing the number of hidden layers or neurons impact model performance?
- **Positive:** Can capture more complex patterns (higher capacity).  
- **Negative:** Increases risk of overfitting, requires more data, and higher computational cost.  
Balance is key → too small = underfitting, too large = overfitting.

---

### 8. What are some signs that your model is overfitting the training data?
- Training accuracy is very high, but test accuracy is much lower.  
- Training loss decreases steadily, but validation loss increases.  
- The model memorizes noise in training data instead of learning general patterns.

---

### 9. Why do we evaluate using precision, recall, and F1-score instead of accuracy alone?
- **Accuracy** can be misleading in imbalanced datasets.  
- **Precision:** Out of predicted positives, how many were correct.  
- **Recall:** Out of actual positives, how many were captured.  
- **F1-score:** Harmonic mean of precision and recall → balances the two.  

Example: In cancer detection, recall is critical (don’t miss positive cases).

---

### 10. How would you improve the model if it performs poorly on the test set?
- Collect more training data.  
- Apply better preprocessing (scaling, feature selection).  
- Use regularization techniques (dropout, L2 penalty).  
- Tune hyperparameters (learning rate, batch size, epochs).  
- Experiment with deeper/wider architectures.  
- Use cross-validation for more reliable evaluation.

---




