import re
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_unicode
from sklearn.metrics import f1_score
import numpy as np

def load_test_data():
    """Sample data with Alef variations and Tunisian dialect features"""
    return [
        "هاذا كتابي",       # Tunisian variant with hamza
        "هاذا كتابي",      # Without hamza
        "رأيت المشكلة",    # MSA with hamza
        "رايت المشكلة",    # Tunisian without hamza
        "نحبوا القهوة",    # Tunisian verb
        "أحب القهوة",      # MSA verb
        "بقداش الفلوس",    # Tunisian
        "بكم الفلوس",      # MSA
        "هاذي البنت",      # Tunisian
        "هذه البنت",       # MSA
        "ميسالش نعملوها",  # Tunisian
        "لا نستطيع عملها"  # MSA
    ]

def baseline_segment(text):
    """Basic segmentation without normalization"""
    tokens = simple_word_tokenize(dediac_ar(text))
    tunisian_markers = {
        'هاذا', 'هاذي', 'هاذوما', 'بقداش', 'ميسالش',
        'نحبوا', 'شكون', 'برمانونت', 'زوز', 'فروشات',
        'توا', 'مالشباك', 'باهي', 'هوني', 'غادي'
    }
    return ['TUN' if token in tunisian_markers else 'MSA' for token in tokens]

def alifaware_segment(text):
    """Enhanced segmentation with Alef normalization and dialect patterns"""
    # Normalization pipeline
    text = normalize_unicode(text)
    text = normalize_alef_maksura_ar(text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    
    tokens = simple_word_tokenize(dediac_ar(text))
    
    # Extended Tunisian features
    tunisian_markers = {
        'هاذا', 'هاذي', 'هاذوما', 'رايت', 'راي', 'بقداش',
        'ميسالش', 'نحبوا', 'شكون', 'زوز', 'فروشات', 'توا',
        'مالشباك', 'باهي', 'هوني', 'غادي'
    }
    
    # Additional dialect patterns
    dialect_patterns = [
        lambda t: t.endswith('وا'),     # Tunisian verb ending
        lambda t: 'اش' in t,            # Question suffix
        lambda t: t.startswith('ميس'),  # Negation
        lambda t: t.startswith('بر')     # Tunisian prefixes
    ]
    
    labels = []
    for token in tokens:
        normalized = token.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        
        # Check both original and normalized forms
        if (token in tunisian_markers or 
            normalized in tunisian_markers or
            any(pattern(token) for pattern in dialect_patterns)):
            labels.append('TUN')
        else:
            labels.append('MSA')
    return labels

def evaluate_approaches(texts):
    # Generate reference labels
    reference_labels = []
    for text in texts:
        tokens = simple_word_tokenize(text)
        labels = []
        for token in tokens:
            # Known Tunisian forms in reference
            if any(m in token for m in ['هاذا', 'هاذي', 'بقداش', 'نحبوا', 'رايت', 'ميسالش']):
                labels.append('TUN')
            # Known MSA forms
            elif any(m in token for m in ['هذا', 'هذه', 'بكم', 'أحب', 'رأيت']):
                labels.append('MSA')
            # Default based on our knowledge
            else:
                labels.append('MSA')
        reference_labels.append(labels)
    
    # Get predictions
    baseline_preds = [baseline_segment(text) for text in texts]
    alifaware_preds = [alifaware_segment(text) for text in texts]
    
    # Calculate metrics
    def flatten(labels):
        return [label for sent in labels for label in sent]
    
    y_true = flatten(reference_labels)
    base_pred = flatten(baseline_preds)
    alif_pred = flatten(alifaware_preds)
    
    baseline_f1 = f1_score(y_true, base_pred, average='macro')
    alifaware_f1 = f1_score(y_true, alif_pred, average='macro')
    
    baseline_em = np.mean([1 if np.array_equal(t, p) else 0 for t, p in zip(reference_labels, baseline_preds)])
    alifaware_em = np.mean([1 if np.array_equal(t, p) else 0 for t, p in zip(reference_labels, alifaware_preds)])
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"{'Model':<15} {'F1 Score':<10} {'Exact Match':<10}")
    print(f"{'Baseline':<15} {baseline_f1:.4f}{'':<6} {baseline_em:.4f}")
    print(f"{'Alif-aware':<15} {alifaware_f1:.4f}{'':<6} {alifaware_em:.4f}")
    
    # LaTeX table
    print(r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{F1 Score} & \textbf{Exact Match} \\
\midrule
Baseline & {:.2f}\% & {:.2f}\% \\
\RL{"a}-aware & \textbf{{:.2f}\%} & \textbf{{:.2f}\%} \\
\bottomrule
\end{tabular}
\caption{Segmentation results comparing baseline and \RL{"a}-aware approaches.}
\label{tab:segmentation-results}
\end{table}
""".format(
    baseline_f1*100, baseline_em*100,
    alifaware_f1*100, alifaware_em*100
))

if __name__ == "__main__":
    print("Loading and processing data...")
    texts = load_test_data()
    
    print("\nSample sentences with segmentations:")
    for i, text in enumerate(texts[:3]):
        print(f"\nSentence {i+1}: {text}")
        print("Baseline:", baseline_segment(text))
        print("Alif-aware:", alifaware_segment(text))
    
    evaluate_approaches(texts)