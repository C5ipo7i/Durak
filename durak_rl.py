from random import shuffle
from random import choice
import sys
import os
import numpy as np
import copy
import pickle

from durak_utils import convert_str_to_1hot,suits_to_num

"""
Possibility Vector is length 53. 52 card choices + picking up the card (as defender)
In the event the defender picks up cards, 53rd option for attacker is pass on giving more cards. 

TODO
Mask possibility vector in the model instead of masking before input
Switch to 37 length vectors
flatten discard pile
add a cards given to opponent vector so that the bot knows what cards the opponent has if any.

"""

class Durak(object):
    def __init__(self,deck,models,funcs,threshold=100,num_players=2,trunk=None,play=False,tournament=False):
        self.deck = deck
        self.num_players = num_players
        self.players = []
        self.models = models
        self.threshold = threshold
        self.play = play
        # if self.play == False: #don't need this because we are using 2 models always
        if tournament == False:
            self.models[-1]._make_predict_function()
            self.models[-2]._make_predict_function()
        else: #Going back to single model for both attack and defense
            # self.models[0][0]._make_predict_function()
            # self.models[0][1]._make_predict_function()
            # self.models[1][0]._make_predict_function()
            # self.models[1][1]._make_predict_function()
            self.models[-1]._make_predict_function()
            self.models[-2]._make_predict_function()

        self.funcs = funcs
        self.results = [0,0]
        self.possibilities = np.arange(53)
        self.epsilon = 0.0001
        self.tournament = tournament
        #reusing previously generated tree
        if trunk == None:
            self.trunk = Tree()
        else:
            self.trunk = trunk
        
    def save_tree(self,tree_path):
        with open(tree_path, 'wb') as handle:
            pickle.dump(self.trunk, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def load_tree(self,tree_path):
        with open(tree_path, 'rb') as handle:
            tree = pickle.load(handle)
        self.trunk = tree
        
    def start_from_state(self,game_state_dict):
        self.play_deck = copy.deepcopy(game_state_dict['deck'])
        self.players = []
        #player1
        hand1 = copy.deepcopy(game_state_dict['hand1'])
        #print(hand1,'hand1')
        hand_1hot1 = convert_str_to_1hot(hand1)
        pl1 = Player(hand1,hand_1hot1,0)
        self.players.append(pl1)
        #player2
        hand2 = copy.deepcopy(game_state_dict['hand2'])
        #print(hand2,'hand2')
        hand_1hot2 = convert_str_to_1hot(hand2)
        pl2 = Player(hand2,hand_1hot2,1)
        self.players.append(pl2)
        #
        trump_card = copy.deepcopy(game_state_dict['trump_card'])
        trump_card_vec = convert_str_to_1hot([trump_card])[0]
        trump_suit = trump_card[1]
        attacking_player = copy.deepcopy(game_state_dict['attacking_player'])
        previous_winner = (False,0)

        discard_pile = copy.deepcopy(game_state_dict['discard_pile'])
        discard_pile_1hot = copy.deepcopy(game_state_dict['discard_pile_1hot'])
        self.game_state = Game_state(trump_card,trump_card_vec,trump_suit,len(self.play_deck),attacking_player,previous_winner,attacking_player,discard_pile=discard_pile,discard_pile_1hot=discard_pile_1hot)
        
    def init_game(self,previous_winner):
        self.play_deck = copy.deepcopy(self.deck)
        shuffle(self.play_deck)
        self.players = []
        for player_id in range(self.num_players):
            #switching to dealing off the front. Then we can append the trump card to the end
            hand = self.play_deck[:6]
            hand_1hot = convert_str_to_1hot(hand)
            del self.play_deck[:6]
            pl = Player(hand,hand_1hot,player_id)
            self.players.append(pl)
        trump_card = self.play_deck[0]
        trump_card_vec = convert_str_to_1hot([trump_card])[0]
        trump_suit = trump_card[1]
#         print(trump_card,'trump card')
        self.play_deck.append(self.play_deck[0])
        self.play_deck.pop(0)
        attacking_player = self.who_starts(previous_winner)
        #Instantiate game state
        self.game_state = Game_state(trump_card,trump_card_vec,trump_suit,len(self.play_deck),attacking_player,previous_winner,attacking_player,discard_pile=[],discard_pile_1hot=[])
        
    def update_game_state(self):
        self.game_state.remaining_cards = len(self.play_deck)
        self.game_state.played_cards = []
        self.game_state.played_cards_1hot = []
        self.game_state.played_card = None
        self.game_state.played_card_1hot = None
        self.game_state.picked_up = False
        
    def int_from_1hot(self,card):
        integer = np.where(card==1)[0]
        
    def print_game_state(self):
        print(self.game_state.remaining_cards,'remaining cards')
        print(self.game_state.played_cards,'played cards')
        print(len(self.game_state.played_cards_1hot),'len played cards 1hot')
        print(self.game_state.played_card,'played card')
        #print(self.game_state.played_card_1hot,'played card 1hot')
        print(self.game_state.discard_pile,'discard_pile')
        #print(self.game_state.discard_pile_1hot.shape,'shape of discard_pile_1hot')
        print(self.game_state.attacking_player,'attacking_player')
        print(self.game_state.defending_player,'defending_player')
        for player in self.players:
            print(len(player.hand),'hand length')
        
    def print_player_attributes(self):
        for player in self.players:
            print(len(player.hand),'hand')
            #print(convert_1hot_to_str(player.hand_1hot),'1hot')
        
    def draw_cards(self,player_id):
        if len(self.play_deck) > 0:
            while len(self.players[player_id].hand) < 6 and len(self.play_deck) > 0:
                self.players[player_id].hand.append(self.play_deck.pop(0))
            self.players[player_id].hand_1hot = convert_str_to_1hot(self.players[player_id].hand)
        
    def who_starts(self,previous_winner):
        if previous_winner[0] == True:
            return previous_winner[1]
        else:
            #check for lowest trump, if no trump then check for lowest card. if tie then split based on suit
            return choice([0,1])
            
    def play_game(self):
        while self.game_state.game_over == False:
            #self.print_game_state() #PRINT STATEMENT
            self.attack()
            if self.game_state.played_card:
#                 print('defending')
                self.defend()
            #check if round is over, else continue
            #self.print_player_attributes()
            self.is_round_over()
            if self.game_state.round_over == True:
                self.update_game_state
                self.is_game_over()
        self.record_outcome()
            
    #to account for giving multiple cards. I'll just check for the taking condition 
    #and then allow the attacker to give up to 6 cards
    def attack(self):
#         print('attacking')
        if self.game_state.played_cards:
            #not first action: We will mask our possibility vector with the like rank cards already played
            #53 because of the possibility of picking up the cards (needs to be consistent for the model)
            possibility_vec = np.zeros((1,53)).flatten()
            possibility_vec[52] = 1
            ranks = []
            [ranks.append(card[0]) for card in self.game_state.played_cards]
#             print(ranks,'ranks')
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                if self.players[self.game_state.attacking_player].hand[i][0] in ranks:
                    location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                    possibility_vec[location] = 1
        else:
            #first action, pick any card Need to locate each card in terms of the entire deck.
            #the possibility vector will be of length 36. with 1s for cards we have and 0s for the rest
#             print('first action')
            #Should just be the hand_1hot vector plus 0
            possibility_vec = np.zeros((1,53)).flatten()
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                possibility_vec[location] = 1
        #inputs = (self.players[player_id].hand_1hot,self.game_state)
        #wrap all options plus game state into something to pass into model
        #if first action can play any card, else has to match
        #decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,0,self)
        decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,0,self)
#         print(decision,'attacking decision')
        if decision == 52:
            #player passed - This is redundent safety
            self.game_state.played_card = None
            self.game_state.played_card_1hot = None
        else:
            #remove card from hand add to played list
            self.remove_card(True,self.game_state.attacking_player,decision)
    
    #defender has the option to defend with any legal card, or take the card
    #Needs current played cards
    def defend(self):
        player_id = self.game_state.defending_player
        possibility_vec = np.zeros((1,53)).flatten()
        attacker_card = self.game_state.played_card
        # print(attacker_card,'attacker_card')
#         [print(card) for card in self.players[self.game_state.defending_player].hand]
#         print('defending players hand')
        #Go through hand and see if any cards can beat it
        for i in range(len(self.players[self.game_state.defending_player].hand)):
            if self.players[player_id].hand[i][0] > attacker_card[0] and self.players[player_id].hand[i][1] == attacker_card[1]            or attacker_card[1] != self.game_state.trump_suit and self.players[player_id].hand[i][1] == self.game_state.trump_suit:
                #possible defend
                location = np.where(self.players[self.game_state.defending_player].hand_1hot[i]==1)[0]
#                 print(location,'defending location')
#                 print(self.players[self.game_state.defending_player].hand[i],'possible card')
                possibility_vec[location] = 1
                i += 1
        #check any cards can defend, otherwise auto pickup
        ### Streamline this later ###
#         if 1 not in possibility_vec:
#             #can't defend, pick up the card
#             picking_up_cards()
#         else:
#             #Add ability to pick up the cards 
#             #pass possibility vec through to model, get action
        ###
        possibility_vec[52] = 1
        decision = self.funcs[self.game_state.defending_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.defending_player].hand_1hot),self.players,self.game_state.defending_player,1,self)
#         print(decision,'defend decision')
        if decision == 52:
            self.picking_up_cards()
        else:
            #chose defend
            #remove that card from hand, add to played list
            self.remove_card(False,self.game_state.defending_player,decision)
            
    def picking_up_cards(self):
        #print('picking up cards')
        #self.print_game_state() ## PRINT STATEMENT
        #add all played cards to hand, check if opponent wants to add any more cards?
        #Check which cards are legal up to the remaining cards in defender's hand
        #self.game_state.played_cards.flatten()
        max_cards = len(self.players[self.game_state.defending_player].hand_1hot)
        
        while max_cards > 0 and len(self.players[self.game_state.attacking_player].hand) > 0:
            possibility_vec = np.zeros((1,53)).flatten()
            possibility_vec[52] = 1
#             if not self.game_state.played_cards:
#                 ranks = self.game_state.played_card[0]
           
            ranks = [card[0] for card in self.game_state.played_cards]
#             print(ranks,'ranks')
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                if self.players[self.game_state.attacking_player].hand[i][0] in ranks:
                    #get 1hot location
                    location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                    possibility_vec[location] = 1
            decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,2,self)
            if decision == 52:
                #to give more cards or not
                break
            else:
                #add card to self.game_state.played_cards
                self.remove_card(False,self.game_state.attacking_player,decision)
                max_cards -= 1
        #add cards to defender's hand
#         print(self.game_state.played_cards,'played cards')
        self.players[self.game_state.defending_player].hand = self.players[self.game_state.defending_player].hand+self.game_state.played_cards
        self.players[self.game_state.defending_player].hand_1hot = self.players[self.game_state.defending_player].hand_1hot+self.game_state.played_cards_1hot
        #clear played cards
        self.game_state.played_cards = []
        self.game_state.played_cards_1hot = []
        self.game_state.played_card = None
        self.game_state.played_card_1hot = None
        self.game_state.picked_up = True
    
    def remove_card(self,attack,player_id,decision):
        #remove card from player's hand
#         print(attack,player_id,decision,'remove_card vars')
        for i in range(len(self.players[player_id].hand)):
            if np.where(self.players[player_id].hand_1hot[i]==1) == decision:
                #add card to played cards list and played card
                if attack == True:
                    self.game_state.played_card = self.players[player_id].hand[i]
                    self.game_state.played_card_1hot = self.players[player_id].hand_1hot[i]
                    self.game_state.played_cards.append(self.game_state.played_card)
                    self.game_state.played_cards_1hot.append(self.game_state.played_card_1hot)
                    #remove card from hand
                    self.players[player_id].hand.pop(i)
                    self.players[player_id].hand_1hot.pop(i)
                else:
                    self.game_state.played_cards.append(self.players[player_id].hand[i])
                    self.game_state.played_cards_1hot.append(self.players[player_id].hand_1hot[i])
                    self.players[player_id].hand.pop(i)
                    self.players[player_id].hand_1hot.pop(i)
                break
        
    def is_round_over(self):
#         print('is round over')
        #is either player out of cards
        #can the attacker continue to play
        #has the defender picked up the cards
        #then check for whether either player needs to draw up to 6
        #check for number of defender's cards
        current_max = len(self.players[self.game_state.attacking_player].hand)
        number_of_attacks = len(self.game_state.played_cards) % 2
        if number_of_attacks == 6 or current_max == 0 or self.game_state.played_card == None:
            #round is over
            self.game_state.round_over = True
            if self.game_state.picked_up == True:
                #attacker draws
                if len(self.play_deck) > 0:
                    self.draw_cards(self.game_state.attacking_player)
                else:
                    pass
            else:
#                 print('successful defense')
                #successful defense, both draw
                self.draw_cards(self.game_state.attacking_player)
                self.draw_cards(self.game_state.defending_player)
                #update attacker and defender
                self.game_state.attacking_player = (self.game_state.attacking_player + 1) % 2
                self.game_state.defending_player = (self.game_state.defending_player + 1) % 2
                #Move played cards to discard pile
                if self.game_state.played_cards:
                    self.game_state.discard_pile.append(self.game_state.played_cards)
                    self.game_state.discard_pile_1hot.append(self.game_state.played_cards_1hot)
            self.update_game_state()
        else:
            #go to additional rounds
            pass
        
    def is_game_over(self):
        for player in self.players:
            if len(player.hand) == 0 and self.game_state.remaining_cards == 0:
                self.game_state.game_over = True
        
    def record_outcome(self):
        if len(self.players[0].hand) == 0 and len(self.players[1].hand) == 0:
            #tie
            self.players[0].outcome = 0
            self.players[1].outcome = 0
            self.game_state.previous_winner = (False,0)
        elif len(self.players[0].hand) == 0:
            self.players[0].outcome = 1
            self.players[1].outcome = -1
            self.results[0] += 1
            self.results[1] -= 1
            self.game_state.previous_winner = (True,0)
        else:
            self.players[0].outcome = -1
            self.players[1].outcome = 1
            self.results[0] -= 1
            self.results[1] += 1
            self.game_state.previous_winner = (True,1)
#         print('game over')
        
#Necessary information
#Trump card. 
#Number of cards remaining in the deck
#Number of cards in hand and opponent's hand
#cards in hand
#Whose turn it is

class Game_state(object):
    def __init__(self,trump_card,trump_card_vec,trump_suit,remaining_cards,attacking_player,previous_winner,first_player,discard_pile=[],discard_pile_1hot=[]):
        self.trump_suit = trump_suit
        self.trump_card = trump_card
        self.trump_num = (suits_to_num([self.trump_card])[0][1])
        self.trump_card_vec = trump_card_vec
        self.remaining_cards = remaining_cards
        self.attacking_player = attacking_player
        self.defending_player = (self.attacking_player + 1) % 2
        self.played_cards = []
        self.played_cards_1hot = []
        self.played_card = None
        self.played_card_1hot = None
        self.discard_pile = discard_pile
        self.discard_pile_1hot = discard_pile_1hot
        self.round_over = False
        self.previous_winner = previous_winner
        self.game_over = False
        self.picked_up = False
        self.first_player = first_player
        self.player_1_history = []
        self.player_2_history = []
        
    def update_player_turn(self):
        self.player_turn = (self.player_turn + 1) % 2
        
class Player(object):
    def __init__(self,hand,hand_1hot,player_id):
        self.hand = hand
        self.hand_1hot = hand_1hot
        self.player_id = player_id
        self.outcome = 0
        self.results = 0