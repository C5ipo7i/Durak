def check_tree():
    #Check tree
    #load tree and be able to traverse from root to leaf
    #Attacker should start with 1 game_state because the initial game_state is always the same
    #The Defender should start with N game_states, where N is number of Attacker choices
    print(durak.game_state.first_root.parent)
    print(durak.game_state.second_root.parent)
    print(durak.game_state.first_root.ev)
    print(durak.game_state.second_root.ev)
    print(len(list(durak.game_state.first_root.child_dict.keys())))
    print(len(list(durak.game_state.second_root.child_dict.keys())))
    root_child_sec = list(durak.game_state.second_root.child_dict.keys())[0]
    readable_child_sec = return_from_hex(root_child_sec)
    print(readable_child_sec,'readable_child_sec')
    root_child_sec2 = list(durak.game_state.second_root.child_dict.keys())[1]
    readable_child_sec2 = return_from_hex(root_child_sec2)
    print(readable_child_sec2,'readable_child_sec2')
    root_child1 = list(durak.game_state.first_root.child_dict.keys())[0]
    readable_child1 = return_from_hex(root_child1)
    print(readable_child1,'readable_child1')
    print(len(list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())),'num actions')
    root_action1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())[0]
    print(root_action1,'root_action1')
    print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action1].ev,'root_action1.ev')
    #readable_action1 = return_from_hex(root_action1)
    #print(readable_action1,'readable_action1')
    root_action2 = list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())[1]
    #readable_action2 = return_from_hex(root_action2)
    #print(readable_action2,'readable_action2')
    print(root_action2,'root_action2')
    print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].ev,'root_action2.ev')
    #game_state after poor choice
    root_gamestate1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict.keys())[0]
    root_action_sub1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].child_dict.keys())[0]
    #pass action
    print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].ev,'root_gamestate1.ev')
    print(len(list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].child_dict.keys())),'game_state children')
    print(root_action_sub1,'root_action_sub1')

    second_root_child = list(durak.game_state.first_root.child_dict[root_child].child_dict.keys())[0]
    third_root_child = durak.game_state.first_root.child_dict[root_child].child_dict[second_root_child]
    #print(list(third_root_child.child_dict.keys()),'third_root_child')
    print(durak.game_state.first_root.child_dict[root_child].child_dict.keys(),'child keys')
    print(durak.game_state.first_root.child_dict[root_child].ev,'child ev')
    print(len(durak.game_state.first_root.child_dict.keys()),'len root keys')
    print(len(durak.game_state.second_root.child_dict.keys()),'len root keys')
    print(durak.trunk.child_dict[0].ev)
    # print(durak.game_state.first_root.child_dict.keys())
    # print(durak.game_state.second_root.child_dict.keys())
    # number_first_nodes = durak.game_state.first_root.count_nodes(durak.game_state.first_root,0)
    # number_second_nodes = durak.game_state.second_root.count_nodes(durak.game_state.second_root,0)

    # print(durak.game_state.first_node.parent)
    # print(durak.game_state.second_node.parent)
    # print(durak.game_state.first_node.ev,'first_node ev')
    # print(durak.game_state.second_node.ev,'second node ev')
    # print(durak.game_state.first_node.child_dict.keys())
    # print(durak.game_state.second_node.child_dict.keys())

    # print(durak.game_state.first_node.parent.parent.parent.ev)
    # print(durak.game_state.second_node.parent.parent.parent.ev)

def check_layers():
    #Check model layer outputs
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

    i = 0
    for layer in model.layers:
        print(layer.name)
        if layer.name == 'dense_379' or layer.name == 'dense_380' or layer.name == 'dense_381':
            config = layer.get_config()
            weights = layer.get_weights() # list of numpy arrays
            print(layer.name)
            print(config,'config')
            print(weights,'weights')
        i+=1
