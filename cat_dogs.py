import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 1. Load Dataset
def load_images_from_folder(folder, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize to uniform size
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append(gray)
            labels.append(label)
    return images, labels

# 2. Dataset Preparation
def prepare_dataset(cat_folder, dog_folder, image_size=(64, 64)):
    print("Loading cat images...")
    cat_images, cat_labels = load_images_from_folder(cat_folder, label=0, image_size=image_size)
    print("Loading dog images...")
    dog_images, dog_labels = load_images_from_folder(dog_folder, label=1, image_size=image_size)

    X = np.array(cat_images + dog_images)  # Combine images
    y = np.array(cat_labels + dog_labels)  # Combine labels
    return X, y

# 3. Feature Extraction
def extract_features(images):
    features = []
    for img in images:
        # Flatten the image (pixel intensities as features)
        features.append(img.flatten())
    return np.array(features)

# 4. Predict Uploaded Image
def predict_uploaded_image(image_path, svm_model, image_size=(64, 64)):
    # Load and preprocess the uploaded image
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = gray.flatten().reshape(1, -1)  # Flatten and reshape for SVM input

        # Predict using the trained SVM
        prediction = svm_model.predict(feature)[0]
        label = "Dog" if prediction == 1 else "Cat"

        return gray, label
    else:
        raise ValueError("Error: Unable to load the image.")

# 5. GUI Implementation
def main():
    # Paths to cat and dog image folders
    CAT_FOLDER = "D:/SHUN/internship/task3/dataset/cat"  # Replace with your path to cat images
    DOG_FOLDER = "D:/SHUN/internship/task3/dataset/dog"  # Replace with your path to dog images

    # Prepare the dataset
    X, y = prepare_dataset(CAT_FOLDER, DOG_FOLDER, image_size=(64, 64))

    # Extract features
    print("Extracting features...")
    X_features = extract_features(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Train an SVM classifier
    print("Training SVM classifier...")
    svm = SVC(kernel='linear')  # Use linear kernel
    svm.fit(X_train, y_train)

    # Predict and Evaluate
    print("Evaluating classifier...")
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # GUI Setup
    root = tk.Tk()
    root.title("Cat vs. Dog Classifier")
    root.state('zoomed')  # Set to full-screen

    # Add scrollable feature
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def show_metrics():
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Accuracy: {accuracy:.2f}\n\n")
        output_text.insert(tk.END, "Classification Report:\n")
        output_text.insert(tk.END, report + "\n")

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks([0, 1], ["Cat", "Dog"])
        plt.yticks([0, 1], ["Cat", "Dog"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=conf_matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def upload_and_predict():
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                # Clear the previous content
                for widget in image_frame.winfo_children():
                    widget.destroy()
                result_label.config(text="")

                # Predict and display
                gray_image, label = predict_uploaded_image(file_path, svm, image_size=(64, 64))

                # Display result
                result_label.config(text=f"Prediction: {label}", font=("Helvetica", 16), foreground="blue")

                # Show uploaded image
                plt.figure(figsize=(5, 5))
                plt.imshow(gray_image, cmap='gray')
                plt.axis('off')

                canvas = FigureCanvasTkAgg(plt.gcf(), master=image_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            except Exception as e:
                messagebox.showerror("Error", str(e))


    # Layout
    frame = ttk.Frame(scrollable_frame, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)

    # Top Frame for Prediction Result
    result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding="10")
    result_frame.pack(fill=tk.X, pady=5)
    result_label = tk.Label(result_frame, text="", font=("Helvetica", 16))
    result_label.pack(pady=5)

    # Main content split
    main_content = ttk.Frame(frame)
    main_content.pack(fill=tk.BOTH, expand=True, pady=5)

    # Left frame for uploaded image
    image_frame = ttk.LabelFrame(main_content, text="Uploaded Image", padding="10")
    image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    # Right frame for confusion matrix
    conf_matrix_frame = ttk.LabelFrame(main_content, text="Confusion Matrix", padding="10")
    conf_matrix_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    # Controls
    metrics_button = ttk.Button(frame, text="Show Metrics", command=show_metrics)
    metrics_button.pack(pady=5)

    upload_button = ttk.Button(frame, text="Upload and Predict", command=upload_and_predict)
    upload_button.pack(pady=5)

    output_text = tk.Text(frame, height=10, width=50, wrap=tk.WORD)
    output_text.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
