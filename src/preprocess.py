"""
Preprocessing del Testo per Analisi Diacronica

Questo modulo gestisce la pipeline di preprocessing del testo:
1. Caricamento dati raw per periodo
2. Normalizzazione (lowercase, rimozione punteggiatura)
3. Tokenizzazione
4. Lemmatizzazione (opzionale)
5. Filtraggio (stopwords, lunghezza token)
6. Salvataggio dati processati

Input: File .txt in data/raw/ (uno per periodo)
Output: File .txt processati in data/processed/
"""

import re
import string
from pathlib import Path
from collections import Counter
import config

# Import condizionali basati su configurazione
if config.NLP_TOOL == 'spacy':
    try:
        import spacy
        nlp = spacy.load(config.SPACY_MODEL)
    except:
        print(f"⚠️  spaCy model '{config.SPACY_MODEL}' non trovato.")
        print(f"   Installare con: python -m spacy download {config.SPACY_MODEL}")
        nlp = None
elif config.NLP_TOOL == 'nltk':
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Verifica risorse NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english')) if config.REMOVE_STOPWORDS else set()
    except ImportError:
        print("⚠️  NLTK non installato. Installare con: pip install nltk")
        raise


class TextPreprocessor:
    """
    Preprocessor per testo con pipeline configurabile.
    """
    
    def __init__(self):
        """Inizializza il preprocessor con configurazioni da config.py"""
        self.nlp_tool = config.NLP_TOOL
        self.use_lemmatization = config.USE_LEMMATIZATION
        self.remove_stopwords = config.REMOVE_STOPWORDS
        self.lowercase = config.LOWERCASE
        self.remove_punctuation = config.REMOVE_PUNCTUATION
        self.remove_numbers = config.REMOVE_NUMBERS
        self.min_length = config.MIN_TOKEN_LENGTH
        self.max_length = config.MAX_TOKEN_LENGTH
        
        if self.nlp_tool == 'spacy' and nlp is None:
            raise ValueError("spaCy non configurato correttamente")
    
    def clean_text(self, text):
        """
        Pulizia base del testo.
        
        Args:
            text: Testo grezzo
        
        Returns:
            Testo pulito
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Rimuovi numeri
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Rimuovi punteggiatura
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalizza spazi
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_spacy(self, text):
        """
        Tokenizzazione con spaCy.
        
        Args:
            text: Testo da tokenizzare
        
        Returns:
            Lista di token (lemmatizzati se configurato)
        """
        doc = nlp(text)
        
        if self.use_lemmatization:
            tokens = [token.lemma_ for token in doc]
        else:
            tokens = [token.text for token in doc]
        
        # Filtra stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in nlp.Defaults.stop_words]
        
        return tokens
    
    def tokenize_nltk(self, text):
        """
        Tokenizzazione con NLTK.
        
        Args:
            text: Testo da tokenizzare
        
        Returns:
            Lista di token (lemmatizzati se configurato)
        """
        tokens = word_tokenize(text)
        
        # Lemmatizzazione
        if self.use_lemmatization:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Filtra stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in stop_words]
        
        return tokens
    
    def filter_tokens(self, tokens):
        """
        Filtra token per lunghezza e validità.
        
        Args:
            tokens: Lista di token
        
        Returns:
            Lista di token filtrati
        """
        filtered = []
        
        for token in tokens:
            # Verifica lunghezza
            if len(token) < self.min_length or len(token) > self.max_length:
                continue
            
            # Verifica che sia alfabetico (opzionale)
            if not token.isalpha():
                continue
            
            filtered.append(token)
        
        return filtered
    
    def process_text(self, text):
        """
        Pipeline completa di preprocessing.
        
        Args:
            text: Testo grezzo
        
        Returns:
            Lista di token processati
        """
        # Step 1: Pulizia
        text = self.clean_text(text)
        
        # Step 2: Tokenizzazione (e lemmatizzazione)
        if self.nlp_tool == 'spacy':
            tokens = self.tokenize_spacy(text)
        elif self.nlp_tool == 'nltk':
            tokens = self.tokenize_nltk(text)
        else:
            # Fallback: split semplice
            tokens = text.split()
        
        # Step 3: Filtraggio
        tokens = self.filter_tokens(tokens)
        
        return tokens


def process_period(period, preprocessor):
    """
    Processa tutti i dati di un periodo.
    
    Args:
        period: Nome del periodo (es. '1900s')
        preprocessor: Istanza di TextPreprocessor
    
    Returns:
        Tuple (tokens processati, statistiche)
    """
    input_file = config.get_period_file_path(period, 'raw')
    
    if not input_file.exists():
        print(f"  ⚠️  File non trovato: {input_file}")
        return None, None
    
    # Carica testo
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocessing
    tokens = preprocessor.process_text(text)
    
    # Statistiche
    stats = {
        'period': period,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'vocab_coverage': len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0,
        'top_10_words': Counter(tokens).most_common(10)
    }
    
    return tokens, stats


def save_processed_data(period, tokens):
    """
    Salva i token processati.
    
    Args:
        period: Nome del periodo
        tokens: Lista di token processati
    """
    output_file = config.get_period_file_path(period, 'processed')
    
    # Salva un token per riga (più facile da processare dopo)
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(f"{token}\n")
    
    print(f"  ✓ Salvato: {output_file}")


def main():
    """
    Main function: preprocessing di tutti i periodi.
    """
    print("="*70)
    print("  Preprocessing Testo - Analisi Diacronica")
    print("="*70)
    print()
    
    print("Configurazione:")
    print(f"  - NLP Tool: {config.NLP_TOOL}")
    print(f"  - Lemmatizzazione: {config.USE_LEMMATIZATION}")
    print(f"  - Rimozione stopwords: {config.REMOVE_STOPWORDS}")
    print(f"  - Lowercase: {config.LOWERCASE}")
    print(f"  - Lunghezza token: {config.MIN_TOKEN_LENGTH}-{config.MAX_TOKEN_LENGTH}")
    print()
    
    # Inizializza preprocessor
    try:
        preprocessor = TextPreprocessor()
    except Exception as e:
        print(f"✗ Errore inizializzazione preprocessor: {e}")
        return
    
    # Crea output directory
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Processa ogni periodo
    all_stats = []
    
    print("Processing periodi...")
    print()
    
    for period in config.PERIODS:
        print(f"[{period}]")
        
        # Preprocessing
        tokens, stats = process_period(period, preprocessor)
        
        if tokens is None:
            continue
        
        # Salva
        save_processed_data(period, tokens)
        
        # Mostra statistiche
        print(f"  Tokens: {stats['total_tokens']:,}")
        print(f"  Parole uniche: {stats['unique_tokens']:,}")
        print(f"  Vocab coverage: {stats['vocab_coverage']:.2%}")
        print(f"  Top 10: {[w for w, c in stats['top_10_words'][:5]]}")
        print()
        
        all_stats.append(stats)
    
    # Statistiche finali
    print("="*70)
    print("  ✅ PREPROCESSING COMPLETATO!")
    print("="*70)
    print()
    
    # Tabella riepilogativa
    print(f"{'Periodo':<10} {'Tokens':<12} {'Unique':<12} {'Coverage':<10}")
    print("-" * 50)
    for stats in all_stats:
        print(f"{stats['period']:<10} {stats['total_tokens']:<12,} "
              f"{stats['unique_tokens']:<12,} {stats['vocab_coverage']:<10.2%}")
    print()
    
    print(f"File processati salvati in: {config.PROCESSED_DATA_DIR}")
    print()


if __name__ == "__main__":
    main()
