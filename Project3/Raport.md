# Your report (Raport)

## Zastosowanie Cross Entropy
Zastosowałem funkcję straty cross entropy poprzez przekształcenie kształtów tensorów: wyjście modelu (logity) z kształtu [batch_size, seq_len, vocab_size] na [batch_size * seq_len, vocab_size] oraz targetów z [batch_size, seq_len] na [batch_size * seq_len], a następnie obliczenie straty między tymi przekształconymi tensorami.

## Generacja tokenów
Generuję tokeny autoregresywnie poprzez wielokrotne pobieranie predykcji modelu, wybieranie następnego tokenu za pomocą dekodowania zachłannego (greedy) lub nucleus sampling, i dołączanie go do sekwencji przy zachowaniu stałego rozmiaru okna kontekstowego.

## Porównanie Nucleus Sampling
Bez nucleus sampling (dekodowanie zachłanne), model konsekwentnie generuje token '3' dla wejścia zaczynającego się od '1', co pokazuje że znalazł lokalne optimum. Z nucleus sampling (top_p=0.5, t=0.1), generacja staje się bardziej zróżnicowana, ponieważ sampling uwzględnia wiele prawdopodobnych tokenów zamiast tylko najbardziej prawdopodobnego. Zapobiega to powtarzającym się wzorcom widocznym w dekodowaniu zachłannym.

## Wpływ parametru Temperature
Parametr temperature kontroluje losowość predykcji. Niższe wartości temperature (jak t=0.1) sprawiają, że rozkład staje się bardziej "spiczasty", faworyzując tokeny o wysokim prawdopodobieństwie. Wyższe wartości temperature (jak t=2.0) spłaszczają rozkład, zwiększając prawdopodobieństwo wyboru tokenów o niskim prawdopodobieństwie, co zwiększa różnorodność ale potencjalnie redukuje spójność.

## Wnioski z wykresu Per Token Accuracy
Wykres dokładności per token pokazuje, że dokładność predykcji poprawia się wraz z większą ilością kontekstu. Wczesne pozycje w sekwencji mają niższą dokładność, ponieważ model ma ograniczony kontekst, podczas gdy późniejsze pozycje korzystają z większej liczby poprzedzających tokenów. Demonstruje to znaczenie kontekstu w modelach językowych opartych na architekturze transformer.
