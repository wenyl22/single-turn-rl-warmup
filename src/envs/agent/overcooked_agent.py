from envs.overcooked_ai_py.mdp.llm_actions import LLMActionSet
from envs.prompts.overcooked import GAME_STATE_PROMPT, MAPPING

class LLMAgent:
    def __init__(self, player_id, layout_name):
        self.player_id = player_id
        self.layout_name = layout_name
        self.player_names = ['Alice', 'Bob']
        # Controls 
        self.DEBUG = False      
        # Enable kitchen counters only for GPT-4, other models cannot handle the complexity
        self.enable_kitchen_counters = True   
        self.single_agent_ablation = False

        self.action_set = LLMActionSet[self.layout_name]
        self.player_actions = []
        self.action_history = []

        # Set other player ID 
        if int(self.player_id) == 0:
            self.other_player_id = 1
        else:
            self.other_player_id = 0

        self.all_actions = []
        for key, value in self.action_set.items():
            if isinstance(value, list):
                self.all_actions.extend(value)
    
    def _get_available_actions(self, state_for_llm):
        # Available Action Constraints
        available_actions = []
        # Check what player is holding 
        if state_for_llm[self.player_id]['held_object'] == "nothing":
            for idx, d in enumerate(state_for_llm['distances']['onion_dispenser']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['onion_dispenser'][idx])

            for idx, d in enumerate(state_for_llm['distances']['plate_dispenser']):
#                print(self.action_set['plate'])
#                print(idx)
                if idx >= len(self.action_set['plate']):
                    raise ValueError("Idx = ", idx, "Length of plate dispenser = ", len(self.action_set['plate']))
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['plate'][idx])
            
            if self.enable_kitchen_counters:
            
                for idx, d in enumerate(state_for_llm['distances']['kitchen_counter']):
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'onion':
                        available_actions.append(self.action_set['kitchen_counter_pick_onion'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_plate'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'soup in plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_soup'][idx])


            if 'storage_counter_pick_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'onion':
                            available_actions.append(self.action_set['storage_counter_pick_onion'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'plate':
                            available_actions.append(self.action_set['storage_counter_pick_plate'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'soup in plate':
                            available_actions.append(self.action_set['storage_counter_pick_soup'][idx])
            
            for idx, d in enumerate(state_for_llm['distances']['gate']):
                if d[0] not in ['infinite']:
                    if state_for_llm['gate_status'][idx] == 'closed':
                        available_actions.append(self.action_set['gate'][idx])
        elif state_for_llm[self.player_id]['held_object'] == 'onion':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooker'][idx])

            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_onion'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_onion'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'plate':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooked_soup'][idx])
            
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_plate' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_plate'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'soup in plate':
            for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['delivery_area'][idx])
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_soup' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_soup'][idx])
        return available_actions + self.action_set['wait'] + self.action_set['collision_avoidance']

    def _correct_dish_to_plate(self, state_for_llm):
        if state_for_llm[self.player_id ]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        
        if state_for_llm[self.other_player_id]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        return state_for_llm

    def _add_distance_info(self, obj_type, idx, d):
        obj_name = MAPPING[obj_type[0]]
        if d[0] == 'infinite' or "blocked" in d[0]:
            description = f"  -{obj_name} {obj_type[0]}{idx}: Position: inaccessible from {self.player_names[int(self.player_id)]}, "
        else:
            description = f"  -{obj_name} {obj_type[0]}{idx}: Position: {d[0]} units away from you({self.player_names[int(self.player_id)]}), "
        if d[1] == 'infinite' or "blocked" in d[1]:
            description += f"inaccessible from {self.player_names[int(self.other_player_id)]}. "
        else:
            description += f"{d[1]} units away from {self.player_names[int(self.other_player_id)]}. "
        return description

    def _add_kitchen_facility_info(self, state_for_llm):
        self.empty_kitchen_counters = []
        self.empty_kitchen_counter_distances = []
        ## cooker states
        description = "-**Cooking Pots**\n"
        for idx, d in enumerate(state_for_llm['distances']['cooker']):
            description += self._add_distance_info('cooker', idx, d)
            description += f" Contains: **{state_for_llm['num_onions_in_pot'][idx]}** onions."
            if state_for_llm['num_onions_in_pot'][idx] == 3:
                description += f"Cook time remaining: {state_for_llm['soup_in_cooker_remaining_time'][idx]}."
            description += "\n"
        ## dispenser states
        description += "\n-**Dispensers**\n"
        for obj_type in ['plate_dispenser', 'onion_dispenser']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                description += self._add_distance_info(obj_type, idx, d) + "\n"
        ## delivery point states
        description += "\n-**Delivery Point**\n"
        for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
            description += self._add_distance_info('delivery_zone', idx, d) + "\n"
        ## kitchen counter states
        if self.enable_kitchen_counters:
            if len(state_for_llm['distances']['kitchen_counter']) > 0:
                description += "\n-**Kitchen Counters with Items**\n"
                flag = 0
                for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                    if state_for_llm['kitchen_counter_objects'][idx] == 'empty':
                        if d[0] in ['infinite'] or 'blocked' in d[0]:
                            self.empty_kitchen_counter_distances.append(float('inf'))
                        else:
                            self.empty_kitchen_counter_distances.append(int(d[0]))
                            self.empty_kitchen_counters.append(f'k{idx}')
                        continue
                    flag = 1
                    description += self._add_distance_info('kitchen_counter', idx, d)
                    description += f"Contains: {state_for_llm['kitchen_counter_objects'][idx]}\n"
                    self.empty_kitchen_counter_distances.append(float('inf'))
                if flag == 0:
                    description += "  -All kitchen counters are empty.\n"
            if len(self.empty_kitchen_counter_distances) > 0:
                description += "\n-**Empty Kitchen Counters**\n"
                closest_kitchen_counter = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                distance_to_closest_kitchen_counter = min(self.empty_kitchen_counter_distances)
                if distance_to_closest_kitchen_counter != float('inf'):
                    description += f'  -Closest empty kitchen counter k{closest_kitchen_counter} is {distance_to_closest_kitchen_counter} units away from {self.player_names[int(self.player_id)]}.\n'
        ## Gate states
        if len(state_for_llm['distances']['gate']) > 0:
            description += "\n-**Gates**\n"
            for idx, d in enumerate(state_for_llm['distances']['gate']):
                description += self._add_distance_info('gate', idx, d)
                if state_for_llm['gate_status'][idx] == 'open':
                    description += f"State: Will stay open for {10 - state_for_llm['gate_open_time'][idx]} time steps.\n"
                else:
                    description += f"State: Closed.\n"
        ## Storage counter states
        if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing'] and len(state_for_llm['distances']['storage_counter']) > 0:
            description += "\n-**Storage Counters with Items**\n"
            for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                if state_for_llm['storage_counter_objects'][idx] == 'empty':
                    continue
                description += self._add_distance_info('storage_counter', idx, d)
                description += f"Contains: {state_for_llm['storage_counter_objects'][idx]}.\n"
        return description

    def _state_to_description(self, state_for_llm, need_history = True):
        state_for_llm = self._correct_dish_to_plate(state_for_llm)
        object_states = self._add_kitchen_facility_info(state_for_llm)
        self.available_actions_list = self._get_available_actions(state_for_llm)
        available_actions = ""
        for i, action in enumerate(self.available_actions_list):
            available_actions += f'{chr(65 + i)}. {action}\n'
        description = GAME_STATE_PROMPT.format(
            my_name=self.player_names[self.player_id],
            my_holding=state_for_llm[self.player_id]['held_object'],
            my_action_history=f"{', '.join(self.action_history[-5:])}\n" if need_history else 'Not available',
            he_name=self.player_names[self.other_player_id],
            he_holding=state_for_llm[self.other_player_id]['held_object'],
            object_states=object_states,
            available_actions=available_actions
        )
        return description
