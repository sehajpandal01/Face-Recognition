import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle
class FaceRecognitionApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.label_encoder = None
        self.img_size = (96, 96)
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    def preprocess_face(self, face_img):
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, self.img_size)
        face_img = face_img.astype('float32') / 255.0
        face_img = face_img.reshape(1, self.img_size[0], self.img_size[1], 1)
        return face_img
    
    def load_dataset(self, dataset_path):
        faces = []
        labels = []
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        detected_faces = self.detect_faces(img)
                        
                        for (x, y, w, h) in detected_faces:
                            face = img[y:y+h, x:x+w]
                            face = cv2.resize(face, self.img_size)
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            faces.append(face)
                            labels.append(person_name)
        
        return np.array(faces), np.array(labels)
    
    def create_model(self, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    def train_model(self, dataset_path, epochs=50):
        faces, labels = self.load_dataset(dataset_path)
        
        if len(faces) == 0:
            print("No faces found in dataset!")
            return
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        categorical_labels = to_categorical(encoded_labels)
        faces = faces.reshape(faces.shape[0], self.img_size[0], self.img_size[1], 1)
        faces = faces.astype('float32') / 255.0
        X_train, X_test, y_train, y_test = train_test_split(
            faces, categorical_labels, test_size=0.2, random_state=42
        )
    
        num_classes = len(self.label_encoder.classes_)
        self.model = self.create_model(num_classes)
        
        print(f"Training model with {num_classes} classes...")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        return history
    def save_model(self, model_path='face_recognition_model.h5', encoder_path='label_encoder.pkl'):
        if self.model:
            self.model.save(model_path)
        if self.label_encoder:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_path='face_recognition_model.h5', encoder_path='label_encoder.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("Model and encoder loaded successfully!")
    
    def predict_face(self, face_img, threshold=0.7):
        if self.model is None or self.label_encoder is None:
            print("Model not loaded!")
            return None, 0
        
        processed_face = self.preprocess_face(face_img)
        predictions = self.model.predict(processed_face, verbose=0)
        confidence = np.max(predictions)
        if confidence > threshold:
            predicted_class = np.argmax(predictions)
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            return predicted_name, confidence
        else:
            return "Unknown", confidence
    
    def recognize_faces_in_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Could not load image!")
            return
        
        faces = self.detect_faces(img)
        
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            name, confidence = self.predict_face(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img
    
    def real_time_recognition(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                name, confidence = self.predict_face(face)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def create_sample_dataset_structure():
    import os
    
    dataset_path = "face_dataset"
    people = ["person1", "person2", "person3"]
    
    for person in people:
        person_path = os.path.join(dataset_path, person)
        os.makedirs(person_path, exist_ok=True)
    
    print(f"Created dataset structure at {dataset_path}")
    print("Add face images for each person in their respective folders")


def main():
    app = FaceRecognitionApp()
    print("=== Step 1: Creating dataset structure ===")
    create_sample_dataset_structure()
    
    dataset_path = "face_dataset"
    if os.path.exists(dataset_path):
        total_images = 0
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                img_count = len([f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{person_name}: {img_count} images")
                total_images += img_count
        
        if total_images > 0:
            print(f"\n=== Step 2: Training model with {total_images} images ===")
            history = app.train_model(dataset_path, epochs=20)
            app.save_model()
            
            print("\n=== Step 3: Testing trained model ===")
            print("Starting real-time recognition. Press 'q' to quit.")
            app.real_time_recognition()
        else:
            print(f"\nNo images found in {dataset_path}")
            print("Please add face images to the person folders:")
            print("face_dataset/person1/ - Add images of person1")
            print("face_dataset/person2/ - Add images of person2")
            print("face_dataset/person3/ - Add images of person3")
            print("Then run the script again.")
    else:
        print(f"Dataset folder {dataset_path} not found!")

if __name__ == "__main__":
    main()

def create_sample_dataset_structure():
    import os
    
    dataset_path = "face_dataset"
    people = ["person1", "person2", "person3"]
    
    for person in people:
        person_path = os.path.join(dataset_path, person)
        os.makedirs(person_path, exist_ok=True)
    
    print(f"Created dataset structure at {dataset_path}")
    print("Add face images for each person in their respective folders")

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
