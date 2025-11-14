import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import random
import json
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CharRNN:
    def __init__(self):
        self.chars = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.model = None
        self.sequence_length = 40
        self.update_count = 0
        self.checkpoint_dir = "training_checkpoints"
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
    def preprocess_text(self, text):
        """Preprocess text and create character mappings"""
        # Create character vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        print(f"Total characters: {len(text)}")
        print(f"Unique characters: {len(self.chars)}")
        print(f"Characters: {''.join(self.chars)}")
        
        return text
    
    def create_sequences(self, text):
        """Create training sequences using sliding window approach"""
        sequences = []
        next_chars = []
        
        for i in range(0, len(text) - self.sequence_length):
            sequences.append(text[i:i + self.sequence_length])
            next_chars.append(text[i + self.sequence_length])
        
        print(f"Number of sequences: {len(sequences)}")
        
        # Convert to one-hot encoded arrays
        X = np.zeros((len(sequences), self.sequence_length, len(self.chars)), dtype=np.bool_)
        y = np.zeros((len(sequences), len(self.chars)), dtype=np.bool_)
        
        for i, sequence in enumerate(sequences):
            for j, char in enumerate(sequence):
                X[i, j, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[next_chars[i]]] = 1
        
        return X, y
    
    def build_model(self):
        """Build the RNN model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.chars))),
            LSTM(128),
            Dense(len(self.chars), activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.01),
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def save_checkpoint(self, epoch, batch):
        """Save model checkpoint and training state"""
        checkpoint_name = f"checkpoint_epoch_{epoch}_batch_{batch}_update_{self.update_count}.h5"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save training state
        training_state = {
            'update_count': self.update_count,
            'epoch': epoch,
            'batch': batch,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'sequence_length': self.sequence_length
        }
        
        state_path = os.path.join(self.checkpoint_dir, f"training_state_{self.update_count}.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(training_state, f)
        
        print(f"✓ Checkpoint saved: {checkpoint_path} (Update {self.update_count})")
    
    def custom_training_callback(self, epoch, batch):
        """Custom callback to save model every 100 updates"""
        self.update_count += 1
        
        if self.update_count % 400 == 0:
            self.save_checkpoint(epoch, batch)
    
    def sample(self, preds, temperature=1.0):
        """Sample from probability distribution with temperature"""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def generate_text(self, seed_text, length=400, temperature=1.0):
        """Generate text using the trained model"""
        generated = seed_text
        
        for i in range(length):
            # Prepare input sequence
            x = np.zeros((1, self.sequence_length, len(self.chars)))
            for j, char in enumerate(seed_text):
                if char in self.char_to_idx:
                    x[0, j, self.char_to_idx[char]] = 1
            
            # Predict next character
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, temperature)
            next_char = self.idx_to_char[next_index]
            
            # Append to generated text and update seed
            generated += next_char
            seed_text = seed_text[1:] + next_char
            
            # Print progress
            if i % 400 == 0:
                print(f"Generated {i}/{length} characters...")
        
        return generated
    
    def train_with_checkpoints(self, text, epochs=50, batch_size=128):
        """Train the model with periodic checkpointing"""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Create sequences
        X, y = self.create_sequences(text)
        
        # Build model
        self.build_model()
        
        # Calculate total batches per epoch
        num_samples = len(X)
        steps_per_epoch = num_samples // batch_size
        if num_samples % batch_size != 0:
            steps_per_epoch += 1
        
        print(f"\nTraining Info:")
        print(f"Total samples: {num_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Checkpoints will be saved every 400 updates")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # Custom training loop to enable per-batch checkpointing
        print("\nStarting training with checkpointing...")
        
        # Convert to TensorFlow dataset for more control
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Training loop
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Split validation data (20%)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(dataset):
                # Train on batch
                batch_loss, batch_accuracy = self.model.train_on_batch(X_batch, y_batch)
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                num_batches += 1
                
                # Call checkpoint callback
                self.custom_training_callback(epoch, batch_idx)
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{steps_per_epoch} - Loss: {batch_loss:.4f} - Accuracy: {batch_accuracy:.4f}")
            
            # Calculate epoch averages
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            # Validation
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Store history
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Training - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"  Total updates: {self.update_count}")
        
        return history
    
    def train(self, text, epochs=50, batch_size=128):
        """Train the model (original method without custom checkpointing)"""
        return self.train_with_checkpoints(text, epochs, batch_size)
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.h5')]
        if not checkpoints:
            print("No checkpoints found!")
            return None
        
        # Sort by update count
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        
        # Load model
        self.model = tf.keras.models.load_model(checkpoint_path)
        
        # Load training state
        update_count = int(latest_checkpoint.split('_')[-1].split('.')[0])
        state_path = os.path.join(self.checkpoint_dir, f"training_state_{update_count}.pkl")
        
        with open(state_path, 'rb') as f:
            training_state = pickle.load(f)
            self.update_count = training_state['update_count']
        
        print(f"Loaded checkpoint: {latest_checkpoint}")
        print(f"Resuming from update {self.update_count}")
        return self.model

def main():
    # Download larger Shakespeare dataset
    import urllib.request
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    shakespeare_text = urllib.request.urlopen(url).read().decode('utf-8')
    
    print(f"Downloaded {len(shakespeare_text)} characters of text")
    
    # Initialize and train the model
    char_rnn = CharRNN()
    
    # Train with checkpointing
    history = char_rnn.train_with_checkpoints(shakespeare_text, epochs=10, batch_size=128)
    
    # Save final model and artifacts
    print("\n" + "="*50)
    print("SAVING FINAL MODEL AND ARTIFACTS")
    print("="*50)
    
    # Save final model
    char_rnn.model.save('trained_model.h5')
    
    # Save character mappings
    mappings = {
        'char_to_idx': char_rnn.char_to_idx,
        'idx_to_char': {k: v for k, v in char_rnn.idx_to_char.items()}
    }
    with open('char_mappings.json', 'w') as f:
        json.dump(mappings, f)
    
    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("Final model and training data saved successfully!")
    print(f"Total training updates: {char_rnn.update_count}")
    print("Checkpoints saved in 'training_checkpoints' directory")
    
    # Generate sample text
    print("\n" + "="*50)
    print("TEXT GENERATION DEMONSTRATION")
    print("="*50)
    
    seed_start = random.randint(0, len(shakespeare_text) - char_rnn.sequence_length - 1)
    seed_text = shakespeare_text[seed_start:seed_start + char_rnn.sequence_length]
    
    print(f"Seed text: '{seed_text}'")
    
    for temperature in [0.5, 1.0, 1.5]:
        print(f"\nGenerated text (temperature={temperature}):")
        print("-" * 40)
        generated = char_rnn.generate_text(seed_text, length=200, temperature=temperature)
        print(generated)
        print("-" * 40)

if __name__ == "__main__":
    main()
