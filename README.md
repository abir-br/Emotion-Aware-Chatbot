# Emotion-Aware Chatbot — VAD-BERT

Système de chatbot capable d’analyser l’état émotionnel d’un utilisateur à partir de son texte et d’adapter automatiquement sa réponse.

Le projet combine le modèle **VAD-BERT**, des règles de décision définies dans un fichier CSV et une **machine à états finis (FSM)** pour automatiser la prise de décision du chatbot.

## Structure du projet

```text
Emotion-Aware-Chatbot/
├── emotion_engine_VAD-BERT.py
├── My_Aggregated_Logic.csv
└── README.md
```

## Description du projet

Le système reçoit un message texte et effectue automatiquement les étapes suivantes :

1. Analyse du texte avec **VAD-BERT**.
2. Extraction des scores **Valence, Arousal et Dominance (VAD)**.
3. Détermination de l’émotion de l’utilisateur.
4. Identification de sa réaction.
5. Sélection automatique d’une action.
6. Mise à jour de l’état du chatbot avec une **Finite State Machine (FSM)**.
7. Génération d’une réponse adaptée.
8. Conservation des derniers échanges dans une mémoire courte.

## Fonctionnement

```text
Message utilisateur
        ↓
     VAD-BERT
        ↓
   Scores VAD
        ↓
Émotion  + Réaction
   ↓          ↓
sadness    rejects help
anger      stays negative
joy        opens up
       ↓
  Système de décision
     ↙       ↘
   CSV        FSM
     ↘       ↙
    Action
       ↓
Réponse automatique
```

### Analyse émotionnelle

Le modèle **VAD-BERT** fournit trois dimensions :

- **Valence** : caractère positif ou négatif de l’état émotionnel.
- **Arousal** : niveau d’activation ou d’intensité émotionnelle.
- **Dominance** : niveau de contrôle associé à l’état émotionnel.

Ces scores sont ensuite transformés en catégories émotionnelles telles que :

- Joie
- Colère
- Tristesse
- Peur
- Surprise
- Dégoût

### Système de décision

Le chatbot combine deux mécanismes :

- **Règles CSV** : sélection d’une action en fonction de l’action actuelle et de la réaction de l’utilisateur.
- **FSM (Finite State Machine)** : gestion de l’évolution de l’état du chatbot et mécanisme de secours lorsqu’aucune règle CSV ne correspond.

Les principaux états sont :

```text
START
SUPPORTING
DEESCALATING
SUGGESTING_PAUSE
END
```

### Mémoire

Le système conserve les **5 derniers échanges** afin de garder un historique court de la conversation.

## Technologies utilisées

- Python
- PyTorch
- Hugging Face Transformers
- VAD-BERT
- Pandas
- NLP
- Finite State Machine (FSM)
- CSV
- JSON

## Installation

Installer les dépendances :

```bash
pip install torch transformers pandas
```

Puis placer les fichiers suivants dans le même répertoire :

```text
emotion_engine_VAD-BERT.py
My_Aggregated_Logic.csv
```

## Exécution

```bash
python emotion_engine_VAD-BERT.py
```

Le programme attend ensuite les messages de l’utilisateur via l’entrée standard.

## Exemple

```text
User: I'm feeling really angry about this situation.
```

Le système analyse automatiquement le message, estime son état émotionnel, détermine sa réaction et choisit une action adaptée.

## Objectifs

Ce projet permet d’explorer :

- l’analyse automatique de texte ;
- l’utilisation d’un modèle Transformer pré-entraîné ;
- la prise de décision automatisée ;
- la combinaison d’un modèle IA avec des règles métier ;
- la génération de réponses adaptées au contexte émotionnel.
