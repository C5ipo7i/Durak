
from keras.layers import Input,dot,multiply,Add,Dense,Activation,Lambda,Flatten,Conv1D,Conv2D,LeakyReLU,Reshape,Concatenate,LSTM,Embedding,Masking
from keras.models import Model, load_model

"""
Could have all levels of abstraction within one model. Or could have them as separate models
Could train how many levels of abstraction are needed. 
Although this seems better to be selected by an outside source.
Like a genetic algorith or EA.

Have a picked_up var for each player that is an feature input

"""
def split_param_game_state(tensor):
    return tensor[:,0]

def split_param_remaining_cards(tensor):
    return tensor[:,1]

def split_param_trump_suit(tensor):
    return tensor[:,2]

def split_param_trump_card(tensor):
    return tensor[:,3:55]

def split_param_hero_hand_length(tensor):
    return tensor[:,55]

def split_param_villain_hand_length(tensor):
    return tensor[:,56]

def split_param_discard_vec(tensor):
    return tensor[:,57:109]

def split_param_played_cards(tensor):
    return tensor[:,109:161]

def split_played_card_emb(tensor):
    return tensor[:,161]

def split_param_hand(tensor):
    return tensor[:,162:214]

def split_possibilities(tensor):
    return tensor[:,214:]

#Embedding splits

def split_param_trump_card_emb(tensor):
    return tensor[:,3]

def split_param_hero_hand_length_emb(tensor):
    return tensor[:,4]

def split_param_villain_hand_length_emb(tensor):
    return tensor[:,5]

def split_param_discard_emb(tensor):
    return tensor[:,6:42]

def split_param_played_cards_emb(tensor):
    return tensor[:,42:54]

def split_param_played_card_emb(tensor):
    return tensor[:,54]

def split_param_hand_emb(tensor):
    return tensor[:,55:91]

def split_possibilities_emb(tensor):
    return tensor[:,91:]

def return_shape(input_shape):
    return (input_shape[1:] + (1,))

#(game_state_vec,remaining_cards_vec,trump_num_vec,trump_card_emb,hero_hand_length,villain_hand_length,discard_emb,played_cards_emb,played_card_emb,hand_emb,possibilities_emb))

def VEmbed_full(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_12_embeddings = Embedding(53,100,input_length=12)
    card_1_embeddings = Embedding(53,100,input_length=1)
    played_cards_only = card_12_embeddings(played_cards)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #played cards
    played_cards_only = Dense(128,activation='relu')(played_cards_only)
    played_cards_only = Dense(1,activation='relu')(played_cards_only)
    print(played_cards_only.shape,'played_cards_only')
    played_cards_only = Reshape((12,))(played_cards_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #discard vec
    discard_vec_only = Dense(128,activation='relu')(discard_vec_only)
    discard_vec_only = Dense(1,activation='relu')(discard_vec_only)
    discard_vec_only = Reshape((36,))(discard_vec_only)
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,trump_card_only,hand,played_card_only,played_cards_only,discard_vec_only,remaining_cards_only,hero_hand_length_only,villain_hand_length_only])
    #print(policy_8.shape,'policy_8')
    policy_8 = Dense(128)(inputs)
    policy_8 = LeakyReLU(alpha)(policy_8)
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    #print(policy_8.shape)
    policy_head = Dense(policy_shape,activation="softmax")(policy_8)
    #add argmax so it directly outputs an index
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(inputs)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_head])
    return model

def VEmbed_ab1(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_1_embeddings = Embedding(53,100,input_length=1)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    played_cards = card_36_embeddings(played_cards)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    X = Dense(128)(inputs)
    X = LeakyReLU(alpha)(X)
    policy = Dense(128)(X)
    policy = LeakyReLU(alpha)(policy)
    policy_head = Dense(policy_shape)(policy)
#     print(policy_head.shape,possibility_mask.shape,'policy, possibility')
#     policy_head = multiply([policy_head,possibility_input])
#     policy_head = Activation("softmax")(policy_head)
    policy_head = Activation("softmax")(policy_head)
    print(policy_head.shape,'policy head')
    #add argmax so it directly outputs an index
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(X)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_head])
    return model


def VEmbed_ab3(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_12_embeddings = Embedding(53,100,input_length=12)
    card_1_embeddings = Embedding(53,100,input_length=1)
    played_cards_only = card_12_embeddings(played_cards)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #played cards
    played_cards_only = Dense(128,activation='relu')(played_cards_only)
    played_cards_only = Dense(1,activation='relu')(played_cards_only)
    print(played_cards_only.shape,'played_cards_only')
    played_cards_only = Reshape((12,))(played_cards_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #policy 1
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand])
    policy = Dense(128)(inputs)
    policy = LeakyReLU(alpha)(policy)
    policy_head1 = Dense(policy_shape)(policy)
    policy_head1 = Activation("softmax")(policy_head1)
    print(policy_head1.shape,'policy head')
    
    #policy 2
    inputs2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    policy2 = Dense(128)(inputs2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy2 = Dense(128)(policy2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy_head2 = Dense(policy_shape)(policy2)
    policy_head2 = Activation("softmax")(policy_head2)
    
    #policy 3
    inputs3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only])
    policy3 = Dense(128)(inputs)
    policy3 = LeakyReLU(alpha)(policy3)
    policy3 = Dense(128)(policy3)
    policy3 = LeakyReLU(alpha)(policy3)
    policy_head3 = Dense(policy_shape)(policy3)
    policy_head3 = Activation("softmax")(policy_head3)
    
    #final policy decision
    policies = Add()([policy_head1,policy_head2,policy_head3])#,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final_head = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(inputs3)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final_head])
    return model

def VEmbed(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_12_embeddings = Embedding(53,100,input_length=12)
    card_1_embeddings = Embedding(53,100,input_length=1)
    played_cards_only = card_12_embeddings(played_cards)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #played cards
    played_cards_only = Dense(128,activation='relu')(played_cards_only)
    played_cards_only = Dense(1,activation='relu')(played_cards_only)
    print(played_cards_only.shape,'played_cards_only')
    played_cards_only = Reshape((12,))(played_cards_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #discard vec
    discard_vec_only = Dense(128,activation='relu')(discard_vec_only)
    discard_vec_only = Dense(1,activation='relu')(discard_vec_only)
    discard_vec_only = Reshape((36,))(discard_vec_only)
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #policy 1
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand])
    policy = Dense(128)(inputs)
    policy = LeakyReLU(alpha)(policy)
    policy_head1 = Dense(policy_shape)(policy)
    policy_head1 = Activation("softmax")(policy_head1)
    print(policy_head1.shape,'policy head')
    
    #policy 2
    inputs2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    policy2 = Dense(128)(inputs2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy2 = Dense(128)(policy2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy_head2 = Dense(policy_shape)(policy2)
    policy_head2 = Activation("softmax")(policy_head2)
    
    #policy 3
    inputs3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only])
    policy3 = Dense(128)(inputs)
    policy3 = LeakyReLU(alpha)(policy3)
    policy3 = Dense(128)(policy3)
    policy3 = LeakyReLU(alpha)(policy3)
    policy_head3 = Dense(policy_shape)(policy3)
    policy_head3 = Activation("softmax")(policy_head3)
    
    #abstract policy4
    policy_4 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,trump_card_only,remaining_cards_only])
    #print(policy_4.shape,'policy_4')
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    #print(policy_4.shape)
    policy_head_4 = Dense(policy_shape,activation="softmax")(policy_4)
    #print(policy_head_4.shape,'policy_head_4')
    
    #abstract policy5
    policy_5 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only,discard_vec_only])
    #print(policy_5.shape,'policy_5')
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    #print(policy_5.shape)
    policy_head_5 = Dense(policy_shape,activation="softmax")(policy_5)
    #print(policy_head_5.shape,'policy_head_5')
    
    #abstract policy6
    policy_6 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only,discard_vec_only,trump_card_only,remaining_cards_only])
    #print(policy_6.shape,'policy_6')
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    #print(policy_6.shape)
    policy_head_6 = Dense(policy_shape,activation="softmax")(policy_6)
    #print(policy_head_6.shape,'policy_head_6')
    
    #abstract policy7
    policy_7 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,discard_vec_only,remaining_cards_only])
    #print(policy_7.shape,'policy_7')
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    #print(policy_7.shape)
    policy_head_7 = Dense(policy_shape,activation="softmax")(policy_7)
    #print(policy_head_7.shape,'policy_head_7')
    
    #abstract policy8
    policy_8 = Concatenate(axis=-1)([game_state_only,trump_suit_only,trump_card_only,hand,played_card_only,played_cards_only,discard_vec_only,remaining_cards_only,hero_hand_length_only,villain_hand_length_only])
    #print(policy_8.shape,'policy_8')
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    #print(policy_8.shape)
    policy_head_8 = Dense(policy_shape,activation="softmax")(policy_8)
    #print(policy_head_8.shape,'policy_head_8')
    
    #final policy decision
    policies = Add()([policy_head1,policy_head2,policy_head3,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])#,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final_head = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(inputs3)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final_head])
    return model

def V1hot(input_shape,policy_shape,alpha,reg_const):
    #Has the following policies
    #all policies look at trump_suit and game_state
    #Policy that looks at hand
    #Policy that looks at hand and played_cards
    #Policy that looks at hand and trump card
    #Policy that looks at hand, discard pile and trump card
    #Policy that looks at hand, discard pile, played_cards and trump card
    #Policy that looks at hand, remaining deck length, played_cards and trump card
    #Policy that looks at hand, discard pile, remaining deck length, played_cards and trump card
    #Policy that looks at everything
    #Input shape = 213 (without picked up cards vec or a villain hand vec)
    #game_state_vec,remaining_cards_vec,trump_num_vec,features.trump_card_vec,hero_hand_length,villain_hand_length,discard_vec,played_cards_vec,hand)
    #0 = game_state_vec (0-2)(1)
    #1 = remaining_cards_vec (0-26)(1)
    #2 = trump_suit_vec (0-3)(1)
    #3-55 = trump_card_vec (52)
    #55-56 = hero_hand_length (0-36)(1)
    #56-57 = villain_hand_length (0-36)(1)
    #57-109 = discard_vec (52)
    #109-161 = played_cards_vec (12)(52)
    #161-213 = hand_vec
    
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_vec)(X_input)
    played_cards = Lambda(split_param_played_cards)(X_input)
    hand_only = Lambda(split_param_hand)(X_input)
    possibility_mask = Lambda(split_possibilities)(X_input)
    #print(hand_only.shape,'hand shape')
    #Embeddings
    game_state_only = Embedding(3,10)(game_state_only)
    remaining_cards_only = Embedding(25,10)(remaining_cards_only)
    trump_suit_only = Embedding(4,10)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20)(villain_hand_length_only)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((10,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #everything
    X = Dense(128)(X_input)
    X = LeakyReLU(alpha)(X)
    X = Dense(128)(X)
    X = LeakyReLU(alpha)(X)
    #print(X.shape,'X')
    
    #Available actions -> emb -> dot with FC -> softmax -> sample/argmax -> action
    
    #full policy
    policy_1 = Dense(128)(X)
    policy_1 = LeakyReLU(alpha)(policy_1)
    policy_1 = Dense(128)(policy_1)
    policy_1 = LeakyReLU(alpha)(policy_1)
    #print(policy_1.shape)
    policy_head_1 = Dense(policy_shape,activation="softmax")(policy_1)
    #print(policy_head_1.shape,'policy_head_1')
    #add argmax so it directly outputs an index
    
    #abstract policy2
    policy_2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only])
    #print(policy_2.shape,'policy_2')
    policy_2 = Dense(128)(policy_2)
    policy_2 = LeakyReLU(alpha)(policy_2)
    policy_2 = Dense(128)(policy_2)
    policy_2 = LeakyReLU(alpha)(policy_2)
    #print(policy_2.shape)
    policy_head_2 = Dense(policy_shape,activation="softmax")(policy_2)
    #print(policy_head_2.shape,'policy_head_2')
    
    #abstract policy3
    policy_3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,played_cards])
    #print(policy_3.shape,'policy_3')
    policy_3 = Dense(128)(policy_3)
    policy_3 = LeakyReLU(alpha)(policy_3)
    policy_3 = Dense(128)(policy_3)
    policy_3 = LeakyReLU(alpha)(policy_3)
    #print(policy_3.shape)
    policy_head_3 = Dense(policy_shape,activation="softmax")(policy_3)
    #print(policy_head_3.shape,'policy_head_3')
    
    #abstract policy4
    policy_4 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,trump_card_only])
    #print(policy_4.shape,'policy_4')
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    #print(policy_4.shape)
    policy_head_4 = Dense(policy_shape,activation="softmax")(policy_4)
    #print(policy_head_4.shape,'policy_head_4')
    
    #abstract policy5
    policy_5 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only])
    #print(policy_5.shape,'policy_5')
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    #print(policy_5.shape)
    policy_head_5 = Dense(policy_shape,activation="softmax")(policy_5)
    #print(policy_head_5.shape,'policy_head_5')
    
    #abstract policy6
    policy_6 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,trump_card_only,remaining_cards_only])
    #print(policy_6.shape,'policy_6')
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    #print(policy_6.shape)
    policy_head_6 = Dense(policy_shape,activation="softmax")(policy_6)
    #print(policy_head_6.shape,'policy_head_6')
    
    #abstract policy7
    policy_7 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,remaining_cards_only])
    #print(policy_7.shape,'policy_7')
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    #print(policy_7.shape)
    policy_head_7 = Dense(policy_shape,activation="softmax")(policy_7)
    #print(policy_head_7.shape,'policy_head_7')
    
    #abstract policy8
    policy_8 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,discard_vec_only,remaining_cards_only])
    #print(policy_8.shape,'policy_8')
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    #print(policy_8.shape)
    policy_head_8 = Dense(policy_shape,activation="softmax")(policy_8)
    #print(policy_head_8.shape,'policy_head_8')
    
    #final policy decision
    policies = Add()([policy_head_1,policy_head_2,policy_head_3,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    value = Concatenate(axis=-1)([X,policies])
    value = Dense(128,activation='relu')(value)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final])
    return model

