# Raport: Multitask Learning dla klasyfikacji i zliczania kształtów geometrycznych

**Autor:** [Twoje Imię]  
**Nr indeksu:** [Twój numer]  
**Data:** [Data]

---

## 1. Wstęp

Celem projektu było zaimplementowanie i porównanie trzech konfiguracji modelu wielozadaniowego:
1. **Tylko klasyfikacja** (λ=0) - identyfikacja 135 konfiguracji kształtów
2. **Tylko regresja** - precyzyjne zliczanie 6 typów kształtów
3. **Multitask** (λ=1.0) - jednoczesna klasyfikacja i regresja

## 2. Analiza danych (EDA)

Zbiór danych zawiera 10 000 obrazów 28×28 pikseli, każdy zawiera dokładnie 10 kształtów dwóch typów.

### 2.1 Statystyki podstawowe
- Liczba obrazów: 10 000
- Podział: 9 000 treningowych, 1 000 walidacyjnych
- Rozmiar obrazu: 28×28 (skala szarości)
- Liczba kształtów na obraz: zawsze 10
- Liczba typów kształtów na obraz: zawsze 2

### 2.2 Rozkład danych (wykresy)

![EDA Results](eda_results.png)

**Wykres A (lewy):** 8 najczęstszych par kształtów. Dominują kombinacje z trójkątami różnych orientacji.

**Wykres B (środkowy):** Rozkład liczby kształtów (tylko niezerowe wartości). Widoczne równomierne rozłożenie od 1 do 9.

**Wykres C (prawy):** 4 przykładowe obrazy ze zbioru. Widać binaryjną naturę danych.

## 3. Architektura modelu

### 3.1 Backbone (wymagany)
