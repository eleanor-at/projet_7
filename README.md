# Projet 07 : Humanités numériques - Extraction de phrases dans des textes anciens sacrés

**Collaborateurs :**
- Laurence Mellerin, chercheuse au laboratoire HISOMA du CNRS
- Irène Gay (P08), philologue

## I. Le Projet

Le projet vise à développer des méthodes algorithmiques pour extraire automatiquement des textes de zones spécifiques de documents anciens numérisés, disponibles en format PDF. Ces documents sont des images de textes sacrés imprimés. L'objectif est de transformer ces images en textes exploitables. Chaque page PDF est convertie en images PNG pour traitement.

## II. Nos Choix Techniques

Au début du projet, nous avions deux options :
1. Créer des critères pour repérer les zones de textes à extraire « à la main » pour ensuite les numériser.
2. Utiliser un outil intelligent que l'on entraînerait pour repérer les zones à extraire.

Notre choix s'est porté sur la première option. La première étape pour repérer le paragraphe qui nous intéresse consiste à repérer son titre (« VERSIO ANTIQUA »). Pour cela, nous avons commencé par faire un traitement statistique sur les lignes et colonnes afin d'isoler les zones contenant les titres. Une fois le titre repéré, il est alors facile d'encadrer le paragraphe. Enfin, on l'extrait avec OpenCV et on le passe dans l'OCR. 

Le programme d'encadrement des paragraphes fonctionne correctement, à l’exception d’une dizaine de pages mal lues. Un code a également été implémenté pour générer des étiquettes contenant les titres et les numéros de chapitres, ce qui permet d’identifier facilement à quel chapitre appartient chaque page traduite — particulièrement utile pour celles où un changement de chapitre intervient. Il ne reste plus qu’à créer les étiquettes des numéros de chapitre et à regrouper les différents modules afin de lancer l’extraction complète.

## III. Nos Outils

- Python (version 3.12.3)
- OpenCV (version 4.12.0)
- Matplotlib (version 3.10.3)
- Numpy (version 2.0.2)
- scipy.signal (version  1.14.1)

## IV. Dernière Version
