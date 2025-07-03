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

Notre choix s'est porté sur la première option. La première étape pour repérer le paragraphe qui nous intéresse consiste à repérer son titre (« VERSIO ANTIQUA »). Pour cela, nous avons commencé par faire un traitement statistique sur les lignes et colonnes afin d'isoler les zones contenant les titres. Pour reconnaître le titre recherché, nous avons créé des étiquettes de ce titre pour les comparer avec les titres repérés via une comparaison cosinus. Une fois le titre repéré, il est alors facile d'encadrer le paragraphe. Une fois cela fait, on l'extrait avec OpenCV et on le passe dans l'OCR.

## III. Nos Outils

- Python
- OpenCV

## IV. Dernière Version
