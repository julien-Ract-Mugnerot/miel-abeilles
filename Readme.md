Problématique:
Une ruche de 101 abeilles dont la population est renouvelée à chaque génération
va être amené à évoluer en mettant en place des méthodes de reproduction et de
mutations qui modifieront le parcours qu'emprête chaque abeille pour aller
récupérer du miel sur les 50 fleurs du champs s'en en ommetre une seule avant de
retourner à la ruche.
Le but étant que les performances de la ruche s'améliorent au fil des générations,
de réaliser un graphique de l'évolution de ces performances et de visualiser
le parcours de la meilleure abeille ainsi que réaliser son arbre généalogique.

Solutions:
La classe Ruche à été réalisée pour simuler la création de la ruche et de la
génération aléatoire de ses prémières abeilles puis de différentes méthodes de
reproduction et de mutations. Plusieurs méthodes de visualisation y sont
associés ainsi qu'un système de sauvegarde du DataFrame des meilleurs ruches
qui pourra être à nouveau importé et réutilisé dans la classe.


J'ai choisis de rendre certains paramètres essentiels de la classe modulables
depuis l'appélation de la classe pour pouvoir les ajuster d'une ruche à l'autre
et même à la suite de la p

Paramètres modifiables de la classe :
steps[1,2*,3,4]
mutations[1,2,3*,4]
def __init__(self, csv_path='Champ de pissenlits et de sauge des pres.csv',\
             quantity = 50, loops=3000, steps = [3, 4], mutations = [1, 4], \
             mutations_starts = [500, 2000, 1, 2000] ,mut_status = True, topXBees = 25):

- Sélection:
Le paramètre quantity permet d'ajuster la quantité d'abeille qui seront éliminées
à chaque nouvelle génération pour en créer de nouvelles,
Le paramètre topXBees permet d'ajuster le nombre de Top abeilles qui seront exclues
de la mutation et pourront se reproduire entre elles

La méthode de sélection qui à été choisie est tout simplement de conserver les
quantity(50 par défaut) de meilleures abeilles et d'éliminer les autres tout
en s'assurant de conserver une variance importante parmis les 50 abeilles
conservées et pas seulement par le biais de la mutation.

La distance cartésienne est utilisée.


- Reproduction:

Quatres méthodes de reproduction ont été réalisées.
La première permettant de sélectionner deux moitiés début et fin du parcours
des deux abeilles parentes
La seconde conservant le milieux des deux séquences.
La troisième sélectionne un morceaux aléatoire(12-15 fleurs) du parcours de deux
abeilles et les assembles au début du parcours de l'abeille naissante.
La quatrième se base sur la troisième méthode mais restitue les séquences
récupérées des parents à la même position dans le parcours de l'enfant dans la
mesure du possible, ces séquences peuvent se chevaucher et les duplicats seront
remis au hasard dans le parcours.

A 50% du parcours la méthode de reproduction change si deux numéros de méthode
ont été fournis dans la variable steps, sous la forme d'une liste.

Le choix final s'est porté sur la troisième méthode qui brasse énormément de
possibilitées. Et qui au milieux du parcours change pour la quatrième méthode
qui est plus fiable pour stabiliser la performance globale de la ruche.


- Mutation:

Chaque abeille hors du TopX, y compris les nouvelles arrivantes de la ruche
peuvent subir plusieurs mutations consécutives différentes sur la même génération.

Quatres mutations proposées :

1. Déplace le début du parcours de l'abeille
2. Inverse le parcours de l'abeille
3. Inverse la position de deux fleurs au hasard
4. Elimine les distances de puis de 550 du parcours.

Les méthodes 1, 2 et 4 ont été conservées après observations de leur performances
et apparaissent à des taux abitraires variables.
La quatrième étant plus irrégulière que les autres, elle apparaît moins souvent.


Conclusion:

Différentes méthodes de reproduction et de sélections ont été envisagées pour
réaliser une ruches en constantes évolutions où les meilleures abeilles ne
deviendront pas toute de pales copies des une des autres après un certain nombre
de simulations et où le potentiel d'atteindre un chemin plus optimal perdure même
après un certain nombre de simulations.

Le meilleur parcours de la meilleure abeille montre une grande progression depuis
l'aléa des premières générations et semble être une option performante, tandis
que l'évolution des performances de la ruche suit la représentation initiale
attendue, et l'arbre généalogique affiche tous les parents bien qu'ils manquent
quelques traits relationnels.


Pistes d'améliorations:

- Ajout d'un déclencheur automatique de mutations quand les résultats commence à
stagner sur une séquence d'un certain nombre de générations.

- Perfectionner l'arbre généalogique en affichant mieux les relations entre les
abeilles parentes aux nombreux enfants avec ces derniers ou en réaffichant les
parents à nouveaux dans l'arbre.

- rajouter l'utilisation temporaire de la seconde méthode de reproduction en
milieu de simulation pour travailler sur le départ de la ruche

- Dictionnaire généalogie sauvegardable en json pour pouvoir recharger l'arbre
généalogique de la meilleure ruche sauvegardée

-Il serait intéressant de créer un dossier saves maintenant qu'il y a trois
dossiers de sauvegarde différents.

- Dernière vérification des commentaires de la fonction.

Notes :
Les commentaires de la fonction sont en anglais
