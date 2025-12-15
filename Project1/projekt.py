"""
Projekt: Multitask Learning dla klasyfikacji i zliczania kształtów geometrycznych
Autor: [Twoje imię/nazwisko]
Data: [Data]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time
from tqdm import tqdm
import random
import warnings
import urllib.request
import zipfile
from collections import defaultdict, Counter
import json

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURACJA
# ============================================================================

# Ustawienie ziaren losowych dla powtarzalności wyników
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# Parametry modelu i uczenia
NUM_CLASSES = 135                 # Liczba klas klasyfikacji
REGRESSION_OUTPUTS = 6            # Liczba kształtów do zliczania
TRAIN_SIZE = 9000                 # Rozmiar zbioru treningowego
VAL_SIZE = 1000                   # Rozmiar zbioru walidacyjnego
BATCH_SIZE_TRAIN = 64            # Rozmiar batcha treningowego
BATCH_SIZE_VAL = 1000            # Rozmiar batcha walidacyjnego
LEARNING_RATE = 1e-3              # Szybkość uczenia
MAX_EPOCHS = 100                  # Maksymalna liczba epok
PATIENCE = 10                     # Cierpliwość dla early stopping

# Wybór urządzenia (GPU jeśli dostępne)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# ============================================================================
# POBRANIE DANYCH
# ============================================================================

def download_data():
    """
    Pobiera i rozpakowuje zbiór danych z repozytorium GitHub.
    Zwraca True jeśli operacja się powiodła, False w przeciwnym razie.
    """
    print("Sprawdzanie dostępności danych...")
    
    # Sprawdzenie czy dane już istnieją
    if os.path.exists('data') and os.path.exists('data/labels.csv'):
        print("✓ Dane już istnieją w lokalnym folderze")
        return True
    
    print("Brak danych lokalnych. Rozpoczynanie pobierania...")
    
    url = "https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip"
    zip_path = "data_gsn.zip"
    
    try:
        # Pobieranie archiwum
        print(f"Pobieranie z {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Pobrano archiwum danych")
        
        # Rozpakowywanie
        print("Rozpakowywanie archiwum...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✓ Rozpakowano dane")
        
        # Weryfikacja
        if os.path.exists('data/labels.csv'):
            print("✓ Dane pomyślnie przygotowane")
            os.remove(zip_path)
            return True
        else:
            print("✗ Błąd: Brak plików po rozpakowaniu")
            return False
            
    except Exception as e:
        print(f"✗ Błąd podczas pobierania: {e}")
        print("\nInstrukcja ręcznego pobrania danych:")
        print("1. Otwórz: https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip")
        print("2. Pobierz plik i wypakuj do folderu 'data/'")
        print("3. Uruchom program ponownie")
        return False

# ============================================================================
# MAPOWANIE KLAS
# ============================================================================

def create_class_mapping():
    """
    Tworzy mapowanie 135 klas na podstawie par kształtów i ich liczebności.
    Każda klasa reprezentuje unikalną kombinację dwóch kształtów.
    """
    shape_names = ['square', 'circle', 'triangle_up', 'triangle_right', 
                   'triangle_down', 'triangle_left']
    
    classes = []
    label_to_class = {}
    class_info = []
    
    class_idx = 0
    # Generowanie wszystkich kombinacji par kształtów
    for i in range(6):
        for j in range(i + 1, 6):
            for count_i in range(1, 10):
                count_j = 10 - count_i
                
                # Tworzenie wektora etykiet
                label = [0] * 6
                label[i] = count_i
                label[j] = count_j
                
                classes.append(label)
                label_to_class[tuple(label)] = class_idx
                
                # Zapis informacji o klasie
                class_info.append({
                    'class_idx': class_idx,
                    'pair': (i, j),
                    'pair_names': (shape_names[i], shape_names[j]),
                    'counts': (count_i, count_j),
                    'vector': label
                })
                class_idx += 1
    
    # Weryfikacja liczby klas
    assert len(classes) == 135, f"Oczekiwano 135 klas, otrzymano {len(classes)}"
    print(f"Utworzono mapowanie {len(classes)} klas")
    
    return classes, label_to_class, class_info

# Globalne mapowania klas
CLASSES_LIST, LABEL_TO_CLASS, CLASS_INFO = create_class_mapping()

# ============================================================================
# DATASET I TRANSFORMACJE
# ============================================================================

class GeometricShapesDataset(Dataset):
    """Dataset zawierający obrazy kształtów geometrycznych z etykietami."""
    
    def __init__(self, root_dir, labels_file, transform=None, is_train=True):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        self.is_train = is_train
        self.label_to_class = LABEL_TO_CLASS
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Wczytanie obrazu
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        image = transforms.ToTensor()(image)
        
        # Wczytanie etykiet
        counts = self.labels_df.iloc[idx, 1:].values.astype(np.float32)
        counts_int = tuple(counts.astype(int))
        
        # Aplikowanie transformacji (tylko dla treningu)
        if self.transform and self.is_train:
            image, counts = self.transform((image, counts))
            counts_int = tuple(counts.astype(int))
        
        # Mapowanie na klasę
        class_label = self.label_to_class.get(counts_int, 0)
        
        return image, class_label, counts


class RandomRotation90:
    """Losowa rotacja obrazu o wielokrotność 90 stopni."""
    
    def __init__(self, p=0.5):
        self.p = p
        # Mapowanie indeksów kształtów po rotacji
        self.rotation_mapping = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 2}
    
    def __call__(self, sample):
        image, counts = sample
        if random.random() < self.p:
            num_rotations = random.randint(1, 3)
            for _ in range(num_rotations):
                image = torch.rot90(image, k=1, dims=[1, 2])
                new_counts = counts.copy()
                for old_idx, new_idx in self.rotation_mapping.items():
                    new_counts[new_idx] = counts[old_idx]
                counts = new_counts
        return image, counts


class RandomFlip:
    """Losowe odbicie poziome lub pionowe."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, counts = sample
        counts = counts.copy()
        
        if random.random() < self.p:
            if random.random() < 0.5:
                # Odbicie poziome
                image = F.hflip(image)
                counts[3], counts[5] = counts[5], counts[3]
            else:
                # Odbicie pionowe
                image = F.vflip(image)
                counts[2], counts[4] = counts[4], counts[2]
        
        return image, counts


class RandomBrightnessContrast:
    """Losowa zmiana jasności i kontrastu."""
    
    def __init__(self, brightness=0.1, contrast=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
    
    def __call__(self, sample):
        image, counts = sample
        if random.random() < self.p:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = F.adjust_brightness(image, brightness_factor)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = F.adjust_contrast(image, contrast_factor)
        return image, counts


class GaussianNoise:
    """Dodawanie szumu gaussowskiego."""
    
    def __init__(self, std=0.05, p=0.5):
        self.std = std
        self.p = p
    
    def __call__(self, sample):
        image, counts = sample
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        return image, counts


class ComposeTransforms:
    """Kompozycja wielu transformacji."""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

# ============================================================================
# MODEL SIECI NEURONOWEJ
# ============================================================================

class MultiTaskModel(nn.Module):
    """
    Model wielozadaniowy do klasyfikacji i regresji.
    Wspólna część konwolucyjna + dwie osobne głowice.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, regression_outputs=REGRESSION_OUTPUTS):
        super(MultiTaskModel, self).__init__()
        
        # Wspólna część konwolucyjna (backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU()
        )
        
        # Głowica klasyfikacyjna
        self.head_cls = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        # Głowica regresyjna (zliczanie)
        self.head_cnt = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, regression_outputs)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        log_probs = self.head_cls(features)
        counts = self.head_cnt(features)
        return log_probs, counts

# ============================================================================
# FUNKCJE STRATY
# ============================================================================

class ClassificationOnlyLoss(nn.Module):
    """Funkcja straty tylko dla zadania klasyfikacji."""
    
    def __init__(self):
        super(ClassificationOnlyLoss, self).__init__()
        self.cls_loss = nn.NLLLoss()
    
    def forward(self, outputs, targets):
        log_probs, _ = outputs
        class_targets, _ = targets
        loss = self.cls_loss(log_probs, class_targets)
        return loss, loss, 0.0


class RegressionOnlyLoss(nn.Module):
    """Funkcja straty tylko dla zadania regresji."""
    
    def __init__(self):
        super(RegressionOnlyLoss, self).__init__()
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(self, outputs, targets):
        _, counts = outputs
        _, count_targets = targets
        loss = self.reg_loss(counts, count_targets)
        return loss, 0.0, loss


class MultiTaskLoss(nn.Module):
    """Połączona funkcja straty dla obu zadań."""
    
    def __init__(self, lambda_cnt=1.0):
        super(MultiTaskLoss, self).__init__()
        self.cls_loss = nn.NLLLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.lambda_cnt = lambda_cnt
    
    def forward(self, outputs, targets):
        log_probs, counts = outputs
        class_targets, count_targets = targets
        
        loss_cls = self.cls_loss(log_probs, class_targets)
        loss_reg = self.reg_loss(counts, count_targets)
        total_loss = loss_cls + self.lambda_cnt * loss_reg
        
        return total_loss, loss_cls, loss_reg

# ============================================================================
# OBLICZANIE METRYK
# ============================================================================

def compute_all_metrics(predictions, targets):
    """
    Oblicza zestaw metryk ewaluacyjnych.
    
    Args:
        predictions: Krotka (predykcje klas, predykcje liczebności)
        targets: Krotka (prawdziwe klasy, prawdziwe liczebności)
    
    Returns:
        Słownik z metrykami
    """
    class_preds, count_preds = predictions
    class_targets, count_targets = targets
    
    # Dokładność klasyfikacji
    accuracy = (class_preds == class_targets).float().mean().item()
    
    # F1-score
    try:
        f1 = f1_score(class_targets.cpu().numpy(), 
                     class_preds.cpu().numpy(), 
                     average='macro')
    except:
        f1 = 0.0
    
    # Dokładność per para kształtów
    pair_correct = defaultdict(int)
    pair_total = defaultdict(int)
    
    for pred, target in zip(class_preds.cpu().numpy(), class_targets.cpu().numpy()):
        if target < len(CLASS_INFO):
            target_pair = CLASS_INFO[target]['pair']
            pred_pair = CLASS_INFO[pred]['pair'] if pred < len(CLASS_INFO) else (-1, -1)
            
            pair_key = tuple(sorted(target_pair))
            pair_total[pair_key] += 1
            
            if tuple(sorted(target_pair)) == tuple(sorted(pred_pair)):
                pair_correct[pair_key] += 1
    
    per_pair_accuracy = {}
    for pair in pair_total:
        per_pair_accuracy[pair] = pair_correct[pair] / pair_total[pair] if pair_total[pair] > 0 else 0.0
    
    # Metryki regresji
    mse_per_class = torch.mean((count_preds - count_targets) ** 2, dim=0)
    rmse_per_class = torch.sqrt(mse_per_class)
    rmse_overall = torch.sqrt(torch.mean((count_preds - count_targets) ** 2))
    
    mae_per_class = torch.mean(torch.abs(count_preds - count_targets), dim=0)
    mae_overall = torch.mean(torch.abs(count_preds - count_targets))
    
    # Nazwy kształtów dla czytelności
    shape_names = ['squares', 'circles', 'up', 'right', 'down', 'left']
    rmse_dict = {shape_names[i]: rmse_per_class[i].item() for i in range(6)}
    mae_dict = {shape_names[i]: mae_per_class[i].item() for i in range(6)}
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'per_pair_accuracy': per_pair_accuracy,
        'rmse_overall': rmse_overall.item(),
        'rmse_per_class': rmse_dict,
        'mae_overall': mae_overall.item(),
        'mae_per_class': mae_dict
    }

# ============================================================================
# PROCEDURY TRENINGU I WALIDACJI
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Przeprowadza jedną epokę treningu."""
    model.train()
    total_loss, cls_loss_total, reg_loss_total = 0, 0, 0
    
    for images, class_labels, count_labels in tqdm(dataloader, desc="Trening"):
        # Przeniesienie danych na odpowiednie urządzenie
        images = images.to(device)
        class_labels = class_labels.to(device)
        count_labels = count_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Obliczenie straty
        loss, cls_loss, reg_loss = criterion(outputs, (class_labels, count_labels))
        
        # Backward pass i aktualizacja wag
        loss.backward()
        optimizer.step()
        
        # Akumulacja statystyk
        total_loss += loss.item()
        cls_loss_total += cls_loss if isinstance(cls_loss, float) else cls_loss.item()
        reg_loss_total += reg_loss if isinstance(reg_loss, float) else reg_loss.item()
    
    # Średnie straty
    avg_total_loss = total_loss / len(dataloader)
    avg_cls_loss = cls_loss_total / len(dataloader)
    avg_reg_loss = reg_loss_total / len(dataloader)
    
    return avg_total_loss, avg_cls_loss, avg_reg_loss


def validate(model, dataloader, criterion, device):
    """Przeprowadza walidację modelu."""
    model.eval()
    total_loss = 0
    all_class_preds, all_class_targets = [], []
    all_count_preds, all_count_targets = [], []
    
    with torch.no_grad():
        for images, class_labels, count_labels in tqdm(dataloader, desc="Walidacja"):
            images = images.to(device)
            class_labels = class_labels.to(device)
            count_labels = count_labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss, _, _ = criterion(outputs, (class_labels, count_labels))
            total_loss += loss.item()
            
            # Zbieranie predykcji
            log_probs, counts = outputs
            class_preds = log_probs.argmax(dim=1)
            
            all_class_preds.append(class_preds.cpu())
            all_class_targets.append(class_labels.cpu())
            all_count_preds.append(counts.cpu())
            all_count_targets.append(count_labels.cpu())
    
    # Agregacja predykcji
    predictions = (torch.cat(all_class_preds), torch.cat(all_count_preds))
    targets = (torch.cat(all_class_targets), torch.cat(all_count_targets))
    
    # Obliczenie metryk
    metrics = compute_all_metrics(predictions, targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics, predictions, targets

# ============================================================================
# EKSPERYMENTY
# ============================================================================

def run_experiment(experiment_name, loss_type='multitask', lambda_cnt=1.0, 
                   augmentations=True, num_epochs=15):
    """
    Przeprowadza kompletny eksperyment treningowy.
    
    Args:
        experiment_name: Nazwa eksperymentu
        loss_type: Typ funkcji straty ('classification', 'regression', 'multitask')
        lambda_cnt: Waga straty regresji w trybie multitask
        augmentations: Czy stosować augmentację danych
        num_epochs: Liczba epok treningu
    
    Returns:
        Słownik z wynikami eksperymentu
    """
    print(f"\n{'='*60}")
    print(f"EKSPERYMENT: {experiment_name}")
    print(f"{'='*60}")
    
    # Przygotowanie transformacji
    if augmentations:
        train_transform = ComposeTransforms([
            RandomFlip(p=0.3),
            RandomRotation90(p=0.3),
            RandomBrightnessContrast(p=0.3),
            GaussianNoise(p=0.2)
        ])
    else:
        train_transform = None
    
    # Wczytanie danych
    full_dataset = GeometricShapesDataset(
        root_dir='data',
        labels_file='data/labels.csv',
        transform=None,
        is_train=True
    )
    
    # Podział na zbiory
    indices = list(range(len(full_dataset)))
    train_indices = indices[:TRAIN_SIZE]
    val_indices = indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    
    # Zastosowanie transformacji do zbioru treningowego
    train_subset = Subset(full_dataset, train_indices)
    
    if train_transform:
        class TransformedSubset(Dataset):
            """Dataset z aplikowanymi transformacjami."""
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            
            def __getitem__(self, idx):
                img, cls, cnt = self.subset[idx]
                if self.transform:
                    img, cnt = self.transform((img, cnt))
                return img, cls, cnt
            
            def __len__(self):
                return len(self.subset)
        
        train_dataset = TransformedSubset(train_subset, train_transform)
    else:
        train_dataset = train_subset
    
    val_dataset = Subset(full_dataset, val_indices)
    
    # DataLoadery
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, 
                           shuffle=False, num_workers=0)
    
    # Inicjalizacja modelu
    model = MultiTaskModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Wybór funkcji straty
    if loss_type == 'classification':
        criterion = ClassificationOnlyLoss()
    elif loss_type == 'regression':
        criterion = RegressionOnlyLoss()
    else:
        criterion = MultiTaskLoss(lambda_cnt=lambda_cnt)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Historia treningu
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_rmse': []
    }
    
    # Trening
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoka {epoch+1}/{num_epochs}")
        
        # Epoka treningu
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Walidacja
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)
        
        # Zapisywanie historii
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_rmse'].append(val_metrics['rmse_overall'])
        
        # Logowanie
        print(f"Strata trening: {train_loss:.4f}")
        print(f"Strata walidacja: {val_loss:.4f}")
        print(f"Dokładność walidacja: {val_metrics['accuracy']:.4f}")
        print(f"RMSE walidacja: {val_metrics['rmse_overall']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Zatrzymanie wczesne po {epoch+1} epokach")
                break
    
    training_time = time.time() - start_time
    
    # Przywrócenie najlepszego modelu
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Ostateczna ewaluacja
    final_val_loss, final_metrics, _, _ = validate(model, val_loader, criterion, device)
    
    return {
        'experiment_name': experiment_name,
        'loss_type': loss_type,
        'lambda_cnt': lambda_cnt if loss_type == 'multitask' else None,
        'training_time': training_time,
        'final_val_loss': final_val_loss,
        'final_metrics': final_metrics,
        'history': history,
        'model': model
    }

# ============================================================================
# WIZUALIZACJA WYNIKÓW
# ============================================================================

def plot_all_results(all_results):
    """
    Generuje wykresy porównawcze dla wszystkich eksperymentów.
    
    Args:
        all_results: Lista wyników eksperymentów
    """
    # 1. Krzywe uczenia
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'classification': 'red', 'regression': 'green', 'multitask': 'blue'}
    
    for results in all_results:
        name = results['experiment_name']
        history = results['history']
        color = colors.get(results['loss_type'], 'black')
        
        axes[0, 0].plot(history['train_loss'], label=name, color=color, linewidth=2)
        axes[0, 1].plot(history['val_loss'], label=name, color=color, linewidth=2)
        axes[1, 0].plot(history['val_accuracy'], label=name, color=color, linewidth=2)
        axes[1, 1].plot(history['val_rmse'], label=name, color=color, linewidth=2)
    
    axes[0, 0].set_title('Strata trening')
    axes[0, 0].set_xlabel('Epoka')
    axes[0, 0].set_ylabel('Strata')
    
    axes[0, 1].set_title('Strata walidacja')
    axes[0, 1].set_xlabel('Epoka')
    axes[0, 1].set_ylabel('Strata')
    
    axes[1, 0].set_title('Dokładność walidacja')
    axes[1, 0].set_xlabel('Epoka')
    axes[1, 0].set_ylabel('Dokładność')
    
    axes[1, 1].set_title('RMSE walidacja')
    axes[1, 1].set_xlabel('Epoka')
    axes[1, 1].set_ylabel('RMSE')
    
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. RMSE i MAE per klasa
    if all_results:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, results in enumerate(all_results):
            metrics = results['final_metrics']
            
            rmse_values = list(metrics['rmse_per_class'].values())
            mae_values = list(metrics['mae_per_class'].values())
            shape_names = list(metrics['rmse_per_class'].keys())
            
            x_pos = np.arange(len(shape_names))
            width = 0.8 / len(all_results)
            
            axes2[0].bar(x_pos + i*width - width*(len(all_results)-1)/2, 
                        rmse_values, width=width, label=results['experiment_name'])
            axes2[1].bar(x_pos + i*width - width*(len(all_results)-1)/2,
                        mae_values, width=width, label=results['experiment_name'])
        
        axes2[0].set_xlabel('Typ kształtu')
        axes2[0].set_ylabel('RMSE')
        axes2[0].set_title('RMSE per klasa')
        axes2[0].set_xticks(x_pos)
        axes2[0].set_xticklabels(shape_names, rotation=45)
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        axes2[1].set_xlabel('Typ kształtu')
        axes2[1].set_ylabel('MAE')
        axes2[1].set_title('MAE per klasa')
        axes2[1].set_xticks(x_pos)
        axes2[1].set_xticklabels(shape_names, rotation=45)
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print("Wygenerowano pliki: learning_curves.png i per_class_metrics.png")

# ============================================================================
# ANALIZA EKSPLORACYJNA DANYCH (EDA)
# ============================================================================

def simple_eda():
    """Przeprowadza podstawową analizę eksploracyjną danych."""
    print("\n" + "="*60)
    print("ANALIZA EKSPLORACYJNA DANYCH (EDA)")
    print("="*60)
    
    # Sprawdzenie danych
    if not os.path.exists('data/labels.csv'):
        print("Brak danych. Próbuję pobrać...")
        success = download_data()
        if not success:
            print("Nie udało się pobrać danych.")
            return
    
    # Wczytanie etykiet
    labels = pd.read_csv('data/labels.csv')
    shape_names = ['squares', 'circles', 'up', 'right', 'down', 'left']
    
    print(f"\n📊 Statystyki zbioru danych:")
    print(f"  Liczba obrazów: {len(labels)}")
    print(f"  Zbiór treningowy: {TRAIN_SIZE}")
    print(f"  Zbiór walidacyjny: {VAL_SIZE}")
    
    # Analiza częstości par kształtów
    shape_pairs = []
    for _, row in labels.iterrows():
        non_zero = [i for i, val in enumerate(row[1:].values) if val > 0]
        if len(non_zero) == 2:
            shape_pairs.append(tuple(sorted(non_zero)))
    
    pair_counts = Counter(shape_pairs)
    
    print(f"  Unikalnych par kształtów: {len(pair_counts)}")
    print("  5 najczęstszych par:")
    for (i, j), count in pair_counts.most_common(5):
        print(f"    {shape_names[i]}-{shape_names[j]}: {count}")
    
    # Wykres 1: Statystyki
    print("\n📈 Generowanie wykresów statystyk...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Częstość par
    top_pairs = pair_counts.most_common(6)
    pair_labels = [f"{shape_names[i]}-{shape_names[j]}" for (i, j), _ in top_pairs]
    pair_values = [count for _, count in top_pairs]
    
    pair_labels = pair_labels[::-1]
    pair_values = pair_values[::-1]
    
    bars = ax1.barh(range(len(pair_labels)), pair_values, color='skyblue', height=0.6)
    ax1.set_yticks(range(len(pair_labels)))
    ax1.set_yticklabels(pair_labels)
    ax1.set_xlabel('Liczba wystąpień')
    ax1.set_title('Najczęstsze pary kształtów')
    ax1.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, pair_values)):
        ax1.text(value + 10, bar.get_y() + bar.get_height()/2,
                f'{value}', ha='left', va='center')
    
    # Rozkład liczby kształtów
    counts_data = labels[shape_names].values
    non_zero_counts = counts_data[counts_data > 0]
    
    if len(non_zero_counts) > 0:
        ax2.hist(non_zero_counts.flatten(), bins=9, range=(1, 10),
                edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Liczba kształtów (gdy występują)')
        ax2.set_ylabel('Częstość')
        ax2.set_title('Rozkład liczby kształtów')
        ax2.set_xticks(range(1, 10))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Brak danych', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Rozkład liczby kształtów')
    
    plt.tight_layout()
    plt.savefig('eda_stats.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Wygenerowano eda_stats.png")
    
    # Wykres 2: Przykładowe obrazy
    print("🖼️  Generowanie przykładowych obrazów...")
    
    fig2 = plt.figure(figsize=(12, 12))
    
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        
        if i < len(labels):
            img_path = f'data/{labels.iloc[i, 0]}'
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('L')
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    # Etykiety
                    counts = labels.iloc[i, 1:].values
                    non_zero = [(shape_names[j], int(counts[j])) 
                               for j in range(6) if counts[j] > 0]
                    
                    if len(non_zero) == 2:
                        ax.set_title(f"Obraz {i+1}: {non_zero[0][0]}:{non_zero[0][1]}, "
                                   f"{non_zero[1][0]}:{non_zero[1][1]}", fontsize=11, pad=10)
                    else:
                        ax.set_title(f"Obraz {i+1}", fontsize=11, pad=10)
                        
                except Exception as e:
                    ax.axis('off')
                    ax.set_title(f"Obraz {i+1}: Błąd", fontsize=11, pad=10)
            else:
                ax.axis('off')
                ax.set_title(f"Obraz {i+1}: Brak pliku", fontsize=11, pad=10)
        else:
            ax.axis('off')
            ax.set_title(f"Obraz {i+1}", fontsize=11, pad=10)
    
    plt.tight_layout()
    plt.savefig('eda_examples.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Wygenerowano eda_examples.png")
    
    # Podsumowanie tekstowe
    with open('eda_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("PODSUMOWANIE ANALIZY DANYCH\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Liczba obrazów: {len(labels)}\n")
        f.write(f"Trening: {TRAIN_SIZE}\n")
        f.write(f"Walidacja: {VAL_SIZE}\n")
        f.write(f"Unikalnych par: {len(pair_counts)}\n\n")
        
        f.write("10 najczęstszych par:\n")
        for (i, j), count in pair_counts.most_common(10):
            f.write(f"  {shape_names[i]}-{shape_names[j]}: {count}\n")
    
    print("✓ Wygenerowano eda_summary.txt")
    print("\n" + "="*60)
    print("EDA ZAKOŃCZONA!")
    print("="*60)

# ============================================================================
# GŁÓWNA FUNKCJA PROGRAMU
# ============================================================================

def main():
    """Główna funkcja zarządzająca całym pipeline'em."""
    print("="*60)
    print("PROJEKT: MULTITASK LEARNING - KSZTAŁTY GEOMETRYCZNE")
    print("="*60)
    
    # 1. Przygotowanie danych
    if not os.path.exists('data/labels.csv'):
        print("\n📥 Pobieranie danych...")
        success = download_data()
        if not success:
            print("\n❌ Nie udało się pobrać danych automatycznie.")
            print("\nWykonaj ręcznie w terminalu:")
            print("wget https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip")
            print("unzip data_gsn.zip")
            print("rm data_gsn.zip")
            print("\nNastępnie uruchom program ponownie.")
            return
    
    # 2. Analiza eksploracyjna danych
    simple_eda()
    
    # 3. Przeprowadzenie eksperymentów
    print("\n" + "="*60)
    print("PRZEPROWADZANIE EKSPERYMENTÓW")
    print("="*60)
    print("UWAGA: Dla pełnego treningu ustaw num_epochs na 100")
    
    all_results = []
    
    # Eksperyment 1: Tylko klasyfikacja
    print("\n[1/3] Tylko klasyfikacja")
    try:
        results1 = run_experiment(
            experiment_name="Tylko klasyfikacja",
            loss_type='classification',
            augmentations=True,
            num_epochs=15  # Skrócone dla demonstracji
        )
        all_results.append(results1)
        print("✓ Zakończono")
    except Exception as e:
        print(f"✗ Błąd: {e}")
        import traceback
        traceback.print_exc()
    
    # Eksperyment 2: Tylko regresja
    print("\n[2/3] Tylko regresja")
    try:
        results2 = run_experiment(
            experiment_name="Tylko regresja",
            loss_type='regression',
            augmentations=True,
            num_epochs=15
        )
        all_results.append(results2)
        print("✓ Zakończono")
    except Exception as e:
        print(f"✗ Błąd: {e}")
    
    # Eksperyment 3: Multitask
    print("\n[3/3] Multitask (λ=1.0)")
    try:
        results3 = run_experiment(
            experiment_name="Multitask (λ=1.0)",
            loss_type='multitask',
            lambda_cnt=1.0,
            augmentations=True,
            num_epochs=15
        )
        all_results.append(results3)
        print("✓ Zakończono")
    except Exception as e:
        print(f"✗ Błąd: {e}")
    
    if all_results:
        # 4. Wizualizacja wyników
        print("\n" + "="*60)
        print("GENEROWANIE WYKRESÓW")
        print("="*60)
        plot_all_results(all_results)
        
        # 5. Zapis wyników
        print("\n" + "="*60)
        print("ZAPISYWANIE WYNIKÓW")
        print("="*60)
        
        # Zapis modeli
        for i, results in enumerate(all_results):
            try:
                torch.save(results['model'].state_dict(), f'model_experiment_{i+1}.pth')
                print(f"✓ Zapisano model {i+1}")
            except:
                print(f"✗ Nie udało się zapisać modelu {i+1}")
        
        # Tabela wyników
        results_table = []
        for results in all_results:
            results_table.append({
                'experiment': results['experiment_name'],
                'training_time': results['training_time'],
                'val_loss': results['final_val_loss'],
                'accuracy': results['final_metrics']['accuracy'],
                'f1_score': results['final_metrics']['f1_score'],
                'rmse_overall': results['final_metrics']['rmse_overall'],
                'mae_overall': results['final_metrics']['mae_overall']
            })
        
        with open('results_summary.json', 'w') as f:
            json.dump(results_table, f, indent=2)
        print("✓ Zapisano results_summary.json")
        
        # Raport tekstowy
        with open('results_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("RAPORT Z EKSPERYMENTÓW\n")
            f.write("="*60 + "\n\n")
            
            for results in all_results:
                f.write(f"{results['experiment_name']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Czas treningu: {results['training_time']:.1f}s\n")
                f.write(f"Strata walidacja: {results['final_val_loss']:.4f}\n")
                f.write(f"Dokładność: {results['final_metrics']['accuracy']:.4f}\n")
                f.write(f"F1-Score: {results['final_metrics']['f1_score']:.4f}\n")
                f.write(f"RMSE: {results['final_metrics']['rmse_overall']:.4f}\n")
                f.write(f"MAE: {results['final_metrics']['mae_overall']:.4f}\n\n")
        
        print("✓ Zapisano results_report.txt")
        
        # Podsumowanie
        print("\n" + "="*60)
        print("PODSUMOWANIE")
        print("="*60)
        
        for results in all_results:
            print(f"\n{results['experiment_name']}:")
            print(f"  Dokładność: {results['final_metrics']['accuracy']:.4f}")
            print(f"  RMSE: {results['final_metrics']['rmse_overall']:.4f}")
            print(f"  Czas treningu: {results['training_time']:.1f}s")
    
    print("\n" + "="*60)
    print("PROGRAM ZAKOŃCZONY")
    print("="*60)
    
    # Lista wygenerowanych plików
    print("\nWygenerowane pliki:")
    files = ['eda_stats.png', 'eda_examples.png', 'eda_summary.txt',
             'learning_curves.png', 'per_class_metrics.png', 
             'results_summary.json', 'results_report.txt']
    
    for i in range(1, 4):
        files.append(f'model_experiment_{i}.pth')
    
    existing_files = [f for f in files if os.path.exists(f)]
    
    if existing_files:
        for file in existing_files:
            size = os.path.getsize(file)
            print(f"  ✓ {file} ({size} bajtów)")
    else:
        print("  ❌ Nie wygenerowano żadnych plików")

# ============================================================================
# URUCHOMIENIE PROGRAMU
# ============================================================================

if __name__ == "__main__":
    main()
