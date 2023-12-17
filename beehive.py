import math
import random as r
import statistics
import os
import copy
import random

import ast
from treelib import Node, Tree
from graphviz import Digraph
from IPython.display import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns



class Honeycomb:
    # Choosed to work on DataFrame Objects

# ---------------------------- Class Def ----------------------------
    # csv_path(str) : path to your csv/xsl file
    # quantity(int) : amount of bees you sacrifice each generation
    # loops(int) : amount of generation you wants to produce
    # steps(list of 2 ints(1-4 range)) : choose reproduction method
    # mutations(list of 1 to 4 ints (1-4 range), order doesn't matter) : choose which mutations you wants to apply
    # mutations_starts(list of ints > 200 as a choosen lim, same dim as mutations, order is related to mut num +1) :
    # : after how many loops will each mutations starts to appear ?

    # mut_status(bool): Mutations activation switch
    # topXBees(int < (100-quantity)) : after sorting them best to worse, how many of your top ranked bees
    # will be able to reproduce and won't be able to mutate, to avoid loosing the best seeds.
    def __init__(self, csv_path='Champ de pissenlits et de sauge des pres.csv',\
                 quantity = 50, loops=3000, steps = [3, 4], mutations = [1, 2, 4], \
                 mutations_starts = [500, 2000, 1, 2500] ,mut_status = True, topXBees = 25):

        self.flower_df = pd.read_excel(csv_path)
        self.flower_pos = list(zip(self.flower_df['x'],self.flower_df['y']))
        self.tree = (500, 500)
        #use flower_pos to setup a clean Dataframe with gen,rank,score
        self.generation = 1
        self.score_table = {}
        self.mutate_table= {}

        self.bee_count = 0
        self.family_tree = {}

        # usefuls vars
        self.loops = loops



        self.quantity = quantity
        self.step = steps



        self.mutations = mutations
        self.mutations_starts = mutations_starts
        self.mut_status = mut_status
        self.topX = topXBees
        self.first_run = True
        self.last_iteration = False
        self.max_xs = 0


        # seed in beta
        self.seed = []

        # process start
        self.honeycomb_df = self.generate_starting_bees()
        self.bs_df = self.honeycomb_df
        self.honeycomb_score()
        self.centralize()

    #test
    def __str__(self):
        return self.honeycomb_df.head()


# ---------------------------- Generate ----------------------------

    def centralize(self):

        # Run beehive's generations, score them,
        # and save generation that overcomes previous ones.
        self.generate_ng_bees(length = 19)
        self.best_score =  min(value for value in self.score_table.values())
        self.mean_score = statistics.mean(value for value in self.score_table.values())
        self.save_df()
        #Plot Results
        self.plot_path(self.honeycomb_df, self.honeycomb_df['score'][0], 0)
        self.plot_evolution()
        if self.first_run:
            self.plot_family_tree()
        self.plot_best_saved_score()
        self.first_run = False

    #generate first 100 bees without tree position
    def generate_starting_bees(self) :
        bees = []
        pos = copy.deepcopy(self.flower_pos)
        for i in range(100) :
            random.shuffle(pos)
            bees.append(list(pos))
            self.seed.append(list(pos))

            #preparing the family tree
            self.bee_count+=1
            self.family_tree[f'{self.bee_count}'] = [0,0]
            self.family_tree

        df = pd.DataFrame({'pos': bees})
        # Add 'Generation' column starting at 1
        df['generation'] = self.generation
        df['rank'] = 100
        df['score'] = 0
        df['id'] = range(1, 101)
        return df

    #generate n+1 bees generation
    def generate_ng_bees(self, length = 10) :
        if not self.first_run:
            self.honeycomb_df = self.honeycomb_df.loc[0:99-self.quantity]
        for gen in range(self.loops):
            #vars
            self.generation +=1
            bees = []
            pos = copy.deepcopy(self.flower_pos)

            # keeping track of progress, could be better with a percentage
            # if gen % 1000 == 0:
            #     print(gen)

            # switching birth method halfway if multiple parameters (only 2 poss atm)
            if gen == self.loops//2:
                self.step.reverse() if len(self.step) >= 2 else self.step

            for i in range(self.quantity) :
                newbee = []
                bee1_id = np.random.randint(0,self.topX)
                bee2_id = np.random.randint(0,self.topX)


                while bee2_id == bee1_id:
                    bee2_id = np.random.randint(0,self.topX)

                #Building family tree beeX_id is actually it's index
                self.bee_count+=1
                self.family_tree[f'{self.bee_count}'] = [\
                                                         self.honeycomb_df['id'][bee1_id], \
                                                         self.honeycomb_df['id'][bee2_id],
                                                        ]

                # First Method Add first half of path, then 2nd
                #
                if self.step[0] == 1:
                    bee1 = self.honeycomb_df['pos'][bee1_id][25:]
                    newbee = copy.deepcopy(bee1)
                    #adding non overlapping flowers from bee2 to newbee
                    for flower in  self.honeycomb_df['pos'][bee2_id][:25]:
                        if flower not in bee1:
                            newbee.append(flower)
                    # adding overlapping flowers in 2nd half of bee 1 and whole of bee 2 to newbee
                    r.shuffle(pos)
                    for flower in pos:
                        if flower not in newbee:
                            newbee.append(flower)
                if self.step[0] == 2:
                    bee1 = self.honeycomb_df['pos'][bee1_id][20:25]
                    newbee = copy.deepcopy(bee1)
                    #adding non overlapping flowers from bee2 to newbee
                    for flower in  self.honeycomb_df['pos'][bee2_id][26:31]:
                        if flower not in bee1:
                            newbee.append(flower)
                    # adding overlapping flowers in 2nd half of bee 1 and whole of bee 2 to newbee
                    r.shuffle(pos)
                    for flower in pos:
                        if flower not in newbee:
                            newbee.append(flower)

                # take a randomely positionned sequence of var lenght (12-15) from 2  TopX bees
                if self.step[0] == 3:
                    rdom =random.randint(0,49-length)
                    bee1 = self.honeycomb_df['pos'][bee1_id][rdom:rdom+length]
                    newbee = copy.deepcopy(bee1)
                    #adding non overlapping flowers from bee2 to newbee
                    rdom2 =random.randint(0,49-length)
                    for flower in  self.honeycomb_df['pos'][bee2_id][rdom2:rdom2+length]:
                        if flower not in bee1:
                            newbee.append(flower)
                    # adding overlapping flowers in 2nd half of bee 1 and whole of bee 2 to newbee
                    r.shuffle(pos)
                    for flower in pos:
                        if flower not in newbee:
                            newbee.append(flower)

                # beta
                if self.step[0] == 4:
#                     length = 10

                    newbee = [(0, 0)] * 50
                    reserve_list   = []
                    dropped_f_bee2 = []

                    rdom =random.randint(0,49-length)
                    newbee[rdom:rdom+length] = self.honeycomb_df['pos'][bee1_id][rdom:rdom+length]
                    #adding non overlapping flowers from bee2 to newbee
                    rdom2 =random.randint(0,49-length)

                    # make sure flowers are put at the same positions in the newbee as it was in bee1-2
                    # or send it to the randomly shuffled reserve list
                    for i, flower in enumerate(self.honeycomb_df['pos'][bee2_id][rdom2:rdom2+length]):
                        if flower not in newbee and newbee[rdom2+i] == (0, 0):
                            ###### peut être -1 ici après le i
                            newbee[rdom2+i] = flower
                        else :
                            if len(dropped_f_bee2) != 0:
                                dropped_f_bee2.append(flower)

                    # adding overlapping flowers in 2nd half of bee 1 and whole of bee 2 to newbee
                    r.shuffle(pos)
                    for flower in pos:
                        if flower not in newbee:
                            reserve_list.append(flower)

                    # aiming to keep the sequence in b2 as close to how it originally was
                    # by joining it unshuffled to the reserved list
                    rd_pos = random.randint(0,49-len(reserve_list)-1)
                    if rd_pos == len(reserve_list)-1:
                        if len(dropped_f_bee2) != 0:
                            reserve_list = reserve_list + dropped_f_bee2
                    elif rd_pos == 0 :
                        if len(dropped_f_bee2) != 0:
                            reserve_list = dropped_f_bee2 + reserve_list
                    else :
                        if len(dropped_f_bee2) != 0:
                            reserve_list = reserve_list[0:rd_pos] + dropped_f_bee2 + \
                                 reserve_list[rd_pos+1:]

                    for i, flower in enumerate(newbee):
                        if flower == (0,0):
                            newbee[i] = reserve_list[0]
                            reserve_list.pop(0)


                # formatting for df
                bees.append({'pos': newbee, 'generation': self.generation, 'rank': 100,\
                             'score' : 111111, 'id' : self.bee_count})
            self.honeycomb_df = self.honeycomb_df.append(bees, ignore_index=True)
            #making sure to keep 100 bees on the last generation
            if gen == self.loops-1:
                self.honeycomb_score(save100=True)
            else :
                self.honeycomb_score()
        return True


    # genotype mutation    reserve : [3]
    def mutate(self, keepers=15) :
        #for bee in self.honeycomb_df[keepers:self.quantity-1]['pos']:
        for bee_index in range(keepers+1,self.quantity-1):
            # the higher p, the less likely the mutation
            # designed as 1/p (1/20)

            # move starting point by one flower
            if 1 in self.mutations and self.generation >= self.mutations_starts[0]:
                p=7
                if np.random.randint(1,p+1) == p:
                    self.honeycomb_df['pos'][bee_index].append(self.honeycomb_df['pos'][bee_index][0])
                    self.honeycomb_df['pos'][bee_index].pop(0)

                    #only will trigger on first type 1 mutate
                    if '1' not in self.mutate_table:
                        self.mutate_table['1'] = {'start': self.generation,'tag': f'Change Starting Point','color': 'purple'}

            # inverse the bee's path
            if 2 in self.mutations and self.generation >= self.mutations_starts[1]:
                p=7
                if np.random.randint(1,p+1) == p:
                    self.honeycomb_df['pos'][bee_index].reverse()
                    if '2' not in self.mutate_table:
                        self.mutate_table['2'] = {'start': self.generation,'tag': 'Reverse Path','color': 'green'}

            #m switch 2 randoms flowers
            if 3 in self.mutations and self.generation >= self.mutations_starts[2]:
                p=50
                flow1_id = np.random.randint(0,50)
                flow2_id = np.random.randint(0,50)

                while flow2_id == flow1_id:
                    flow2_id = np.random.randint(0,50)

                self.honeycomb_df['pos'][bee_index][flow1_id], self.honeycomb_df['pos'][bee_index][flow2_id] = \
                self.honeycomb_df['pos'][bee_index][flow2_id], self.honeycomb_df['pos'][bee_index][flow1_id]
                if '3' not in self.mutate_table:
                        self.mutate_table['3'] = {'start': self.generation,'tag': 'Switch Two Flowers','color': 'blue'}

                break
#             # realize on sight that two of those flowers are 2 meters appart

#             # NOT A MUTATION BUT ADD COPYING TOP 3 TO NEWBEES
#             if 5 in mutations:
#                 p=1/10
#                 break

            # copies a top bee and tries to avoid long travels
            # by switching flower positions of n+1 flowers
            # with a random flower except first 5 and last 5
            if 4 in self.mutations and self.generation >= self.mutations_starts[3]:
                p = 7
                if np.random.randint(1,p+1) == p:
                    df = pd.DataFrame(columns=['id', 'distance'])

                    # find the id of the biggest distance traveled
                    # between 2 flowers n and n+1

                    self.quantity-1

                    # mutating bee copies a top 5 bees
                    # tries to eliminate really long deplacements
                    top_bee_index = random.randint(0,6)
                    self.honeycomb_df['pos'][bee_index] = self.honeycomb_df['pos'][top_bee_index]
                    for i in range(len(self.honeycomb_df['pos'][bee_index])-1):
                        df.loc[len(df)] = [i, self.distance(self.honeycomb_df['pos'][bee_index], i, i+1)]

                    bee_indexs = df[df['distance']> 450].sort_values(by='distance', ascending=False)['id']

                    for flow1_id in bee_indexs:
                        #modifying pos of n+1 flower, to hope for
                        flow1_id +=1
                        # a voir si on évite pas entre 4,45 ou 0, 50
                        flow2_id = np.random.randint(4,45)

                        while flow2_id == flow1_id or flow2_id == flow1_id-1:
                            flow2_id = np.random.randint(0,50)

                        self.honeycomb_df['pos'][flow1_id], self.honeycomb_df['pos'][flow2_id] = \
                        self.honeycomb_df['pos'][flow2_id], self.honeycomb_df['pos'][flow1_id]

                    if '4' not in self.mutate_table:
                        self.mutate_table['4'] = {'start': self.generation,'tag': 'Cut Longest travel','color': 'yellow'}
            break

# ---------------------------- Calculus ----------------------------

    def honeycomb_score(self, save100 = False):
        # give a score to all 100 bees's paths
        for i in range(100):
            self.honeycomb_df.loc[i, 'score'] = self.bee_score(self.honeycomb_df['pos'][i])

        # cut the last 50 bees, and clean DataFrame
        self.honeycomb_df = self.honeycomb_df.sort_values(by='score').reset_index()
        self.honeycomb_df['rank'] = range(1, 101)

        # MUTATE
        if  self.loops >= 200 and self.mut_status:
            self.mutate()

        # save mean score in a table for later plots
        self.score_table[f"{self.generation}"] = self.honeycomb_df['score'].mean()

        #preparing generations graph dimensions
        if self.generation == 1:
            self.max_xs = self.honeycomb_df['score'].mean()

        # saving 100 if asked to
        if save100 != True:
            self.honeycomb_df = self.honeycomb_df[0:100-self.quantity]

        #dropping unwanted column
        self.honeycomb_df.drop(columns='index', inplace = True)

        # CALL GENERATE 50 NEWBEES
#         self.plot_path(self.honeycomb_df)
        return True


    # score will be higher the lesser the total distance the bee went throught
    # range(52) since we're adding leaving/returning to the tree
    def bee_score(self, single_bee_list):
        #add tree to the bee's path and score bee's path
        return sum([self.distance([(500, 500)] + single_bee_list+ [(500, 500)], i, i+1) for i in range(51)])


    # take the bee's full path, index and index +=1 to calculate distance between those 2 points
    def distance(self, flower_list, flower_1, flower_2):
        return math.sqrt((flower_list[flower_2][1] - flower_list[flower_1][1])**2 + \
                         (flower_list[flower_2][0] - flower_list[flower_1][0])**2)



# ---------------------------- Save & Load ----------------------------

    # save df if mean score is the best we've seen yet
    def save_df(self, force_it=False):
        if force_it:
            self.honeycomb_df.to_csv('data.csv', index=False)
            return True

        files = os.listdir('bee_saves')
        #check if there are already saves, and saves df
        if files:
            clean_files = [int(f.replace('.csv', '')) for f in files]

            if self.best_score < min(clean_files):
                print(f"New Best Score : {self.best_score}")
                self.honeycomb_df.to_csv(f"bee_saves/{int(self.best_score)}.csv", index=False)
        else:
            print('empty saves ??')
            self.honeycomb_df.to_csv(f"bee_saves/{int(self.best_score)}.csv", index=False)

    # Used for saved csv only
    def plot_best_saved_score(self):


        files = os.listdir('bee_saves')

        #check if there are already saves, and saves df
        if files:
            clean_files = [int(f.replace('.csv', '')) for f in files]

            to_plot_df = pd.read_csv(f'bee_saves/{min(clean_files)}.csv')
            to_plot_df['pos'] = to_plot_df['pos'].apply(lambda x: ast.literal_eval(x))
            self.plot_path(to_plot_df, to_plot_df['score'].min())
        else :
            print('no saves available !')

    # retrain best if called without params, file name without csv
    def restart_training(self, file_name = False, loops = 10000):
        # files needs to be updated at every of those methods starts.
        files = os.listdir('bee_saves')
        if files:
            clean_files = [int(f.replace('.csv', '')) for f in files]
            if file_name == False:
                df = pd.read_csv(f'bee_saves/{min(clean_files)}.csv')
            else :
                df = pd.read_csv(f'bee_saves/{file_name}.csv')

            df['pos'] = df['pos'].apply(lambda x: ast.literal_eval(x))
            self.honeycomb_df = df
            self.loops = loops
            self.last_iteration = False
            #making sure to plot the correct best score of the reloaded hive
#             self.best_score = self.honeycomb_df['score'][0]

            self.centralize()

# ---------------------------- Vizualisation ----------------------------

    def plot_points(self, df):
        plt.scatter(df['x'], df['y'],marker='o', color ="orange", zorder=1)
        self.plot_tree()
        plt.show()

    def plot_path(self, df, score, i=0):
        temp_df = [(500, 500)] + df.loc[i, 'pos'] + [(500, 500)]

        x, y = zip(*temp_df)
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='', label='Points')


        plt.plot(x, y,marker='o', color ="orange", zorder=1)
        plt.xlabel('x')
        plt.ylabel('y')
        if int(self.best_score) != int(score):
            plt.title(f"Record Fitness Score: {int(score)}")
        else :
            plt.title(f'Most Performing Bee|Score: {int(score)}')
        self.plot_tree()
        plt.savefig("plot_saves/best_bee_path.png")
        plt.show()

    def plot_tree(self):
        plt.scatter(500, 500, color='green', marker='^', s=300, zorder=2)
        plt.scatter(500, 500, color='green', marker='1', s=300, zorder=2)
        plt.scatter(500, 450, color='green', marker='1', s=300, zorder=2)
        plt.scatter(500, 480, color='magenta', marker='p', s=20, zorder=10)
        plt.scatter(500, 550, color='gold', marker='*', s=80, zorder=4)
        plt.scatter(500, 550, color='black', marker='*', s=113, zorder=3)
        plt.scatter(476, 520, color='red', marker='o', s=20, zorder=3)
        plt.scatter(524, 520, color='red', marker='o', s=20, zorder=4)
        plt.scatter(500, 480, color='black', marker='v', s=85, zorder=4)
        plt.scatter(511, 480, color='white', marker='^', s=45, zorder=4)
        plt.scatter(489, 480, color='white', marker='^', s=45, zorder=4)


    def plot_evolution(self):
        generations = list(map(int, self.score_table.keys()))
        performance = list(self.score_table.values())
        plt.figure()
        plt.plot(generations, performance, color ="orange")  # Adjust marker as needed
        plt.xlabel('Generation')
        plt.ylabel('Mean Performance')
        plt.title('Performance vs Generation')
        plt.grid(True)

        self.plot_mutations()

        # checking if legend is wanted for mutation
        # and then check if mutation actually took place in this simulation
        if self.mut_status and self.mutate_table:
            plt.legend()

        plt.savefig("plot_saves/hive_score.png")
        plt.show()

    def plot_mutations(self):
        legends = []
        for mut in self.mutate_table:

            #check if mutation is activated and actually happened
            if not self.mutate_table[mut]:
                continue

            if self.mut_status :
                plt.axvline(x=self.mutate_table[mut]['start'],\
                            color=self.mutate_table[mut]['color'], \
                            label=self.mutate_table[mut]['tag'])
            else :
                plt.axvline(x=self.mutate_table[mut]['start'],\
                            color=self.mutate_table[mut]['color'])

#             legends.append(mpatches.Patch(color = self.mutate_table[mut]['color']\
#                                           , label=self.mutate_table[mut]['tag']))
#         plt.legend(handles=[legends])


    def plot_family_tree(self, printing=True):
        tree = Tree()
        target_bee = self.honeycomb_df['id'][0]
        parents_list = []
        #loop number of generation to display
        for i in range(1,6):
            if i == 1 :
                parents_list.extend(self.family_tree[f"{target_bee}"])
                tree.create_node(f"{target_bee}", f"{target_bee}")
                son_list = [target_bee]
            # loop
            temp_list = []
            #loop for each parents the current generation
            for j in parents_list:
                # leave the loop if 0(Queen) is in parents, proceed to graph
                if 0 in parents_list:
                    break

                temp_list.extend(self.family_tree[f"{j}"])
                # reversing a parent sons logic since we're starting from a child
                # to check
                if f"{j}" not in tree:
                    tree.create_node(f"{j}", f"{j}", \
                            parent=f"{int(son_list[parents_list.index(int(j))//2])}")
            son_list = parents_list
            parents_list = temp_list
            temp_list = []

        #tree.show()

        # Creating a Digraph object from graphviz and adding nodes and edges
        dot = Digraph()
        for node in tree.all_nodes():
            dot.node(node.identifier, node.tag)
            if node.is_root():
                continue  # Skip adding edge for the root node
            parent_node = tree.get_node(node.predecessor(tree.identifier))
            dot.edge(node.identifier, parent_node.identifier)
        # Display the graph
        # Check if the directory exists; if not, create it
        directory = '/home/julienrm/code/plateforme/La_plateforme_Ract-Mugnerot_Julien/bees/tree_plots'
        # Save the file in the specified folder
        dot.render('tree_plot', format='png', directory=directory, view=False)

        # ideally directly print image in notebook,sadly this command only
        # display the plot when written last in a cell
        if printing == True:
            Image(filename='tree_plots/tree_plot.png')
