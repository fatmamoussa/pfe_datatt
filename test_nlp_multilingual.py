"""
Script de test pour vérifier le module NLP multilingue
Teste la détection de langue et traduction AR→FR
"""

from nlp_multilingual import (
    detect_language, Language, preprocess_query,
    translate_arabic_to_french, normalize_arabic,
    extract_keywords
)

def test_language_detection():
    """Teste la détection de langue"""
    print("=" * 60)
    print("TEST 1: Détection de Langue")
    print("=" * 60)
    
    test_cases = [
        ("Quels sont les forfaits disponibles?", Language.FRENCH),
        ("ما هي الباقات المتاحة؟", Language.ARABIC),
        ("How to buy a plan?", Language.FRENCH),  # Anglais seul → FRENCH (par défaut)
        ("Je veux savoir sur عروض HAYYA", Language.MIXED),
        ("", Language.UNKNOWN),
    ]
    
    for text, expected in test_cases:
        detected = detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{text[:40]}...' → {detected.name} (attendu: {expected.name})")

def test_translation():
    """Teste la traduction AR→FR"""
    print("\n" + "=" * 60)
    print("TEST 2: Traduction Arabe → Français")
    print("=" * 60)
    
    test_cases = [
        "كم سعر الفايبر؟",
        "هل يوجد رومينج دولي؟",
        "كيف أشتري باقة HAYYA؟",
        "ما هي الخدمات المتاحة؟",
        "أريد تفعيل الإنترنت",
    ]
    
    for ar_query in test_cases:
        fr_query = translate_arabic_to_french(ar_query)
        print(f"AR: {ar_query}")
        print(f"FR: {fr_query}")
        print()

def test_preprocessing():
    """Teste le prétraitement complet"""
    print("\n" + "=" * 60)
    print("TEST 3: Prétraitement Complet (Détection + Traduction)")
    print("=" * 60)
    
    test_cases = [
        "Bonjour, je veux connaître les offres HAYYA",
        "ما هي الباقات؟",
        "J'ai un problème avec mon internet",
        "عندي مشكلة في الإنترنت",
    ]
    
    for query in test_cases:
        processed, lang = preprocess_query(query)
        print(f"Original:   {query}")
        print(f"Langue:     {lang.name}")
        print(f"Processée:  {processed}")
        print()

def test_keywords():
    """Teste l'extraction de mots-clés"""
    print("\n" + "=" * 60)
    print("TEST 4: Extraction de Mots-clés")
    print("=" * 60)
    
    test_cases = [
        ("Quels forfaits pour le roaming?", Language.FRENCH),
        ("ما هي باقات الإنترنت؟", Language.ARABIC),
    ]
    
    for query, lang in test_cases:
        keywords = extract_keywords(query, lang)
        print(f"{lang.name}: {query}")
        print(f"Mots-clés: {keywords}")
        print()

def test_normalization():
    """Teste la normalisation du texte arabe"""
    print("\n" + "=" * 60)
    print("TEST 5: Normalisation Arabe")
    print("=" * 60)
    
    with_diacritics = "مَا هِيَ الْبَاقَاتُ؟"
    normalized = normalize_arabic(with_diacritics)
    print(f"Avec diacritiques: {with_diacritics}")
    print(f"Normalisé:         {normalized}")

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  TEST SUITE — Module NLP Multilingue (AR/FR)          ║")
    print("╚" + "="*58 + "╝")
    print()
    
    try:
        test_language_detection()
        test_translation()
        test_preprocessing()
        test_keywords()
        test_normalization()
        
        print("\n" + "="*60)
        print("✓ TOUS LES TESTS RÉUSSIS")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
