import datetime
import os.path
from copy import deepcopy
from typing import *

from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.llms.llm_negotiation import Negotiation
from src.llms.llm_primitives import GPT, DeepInfra, Gemini
from src.utils.grid import Grid
from src.utils.movement import *
from src.utils.utils import Utils
from src.llms.llm_primitives import LLamaAPI

class LLMMoveGen:
    def __init__(
        self,
        scenario,
        agent_ids,
        valid_pos,
        agent_starting_pos,
        agent_goal_pos,
        model_name,
        env_change_rate,
        write_csv=False,
        path_name="llm",
        write_conversation=False,
    ):
        self.agent_ids = agent_ids
        self.valid_pos = valid_pos
        self.agent_starting_pos = agent_starting_pos
        self.agent_goal_pos = agent_goal_pos
        self.agent_pos = deepcopy(agent_starting_pos)
        self.model = model_name
        self.write_csv = write_csv
        self.scenario = scenario
        self.env_change_rate = env_change_rate
        self.path_name = path_name
        self.write_conversation = write_conversation
        self.conversation_file = self._make_conversation_file()

        assert self.valid_pos == self.scenario.valid_pos

        # print(set(self.agent_ids))
        # print(set(self.agent_starting_pos.keys()))
        # print(set(self.agent_goal_pos.keys()))
        assert set(self.agent_ids) == set(self.agent_starting_pos.keys()) and set(
            self.agent_ids
        ) == set(self.agent_goal_pos.keys())

    def _move(self, name: str, action: str) -> Tuple[Tuple[int, int], Optional[str], bool]:
        """Moves the agent

        Args:
          name: the name of the agent to move
          action: the action to perform

        Returns: the agent's new location
        """

        inv_act_map = {
            "@FORWARD": "@NORTH",
            "@BACKWARD": "@SOUTH",
            "@RIGHT": "@EAST",
            "@LEFT": "@WEST",
            "@WAIT": "@WAIT",
        }

        name = name.lower()
        other_name = "bob" if name == "alice" else "alice"

        dx = 0
        dy = 0

        old_pos = self.agent_pos[name]

        if action_to_string(Action.FORWARD) in action:
            dy = 1
        elif action_to_string(Action.BACKWARD) in action:
            dy = -1
        elif action_to_string(Action.LEFT) in action:
            dx = -1
        elif action_to_string(Action.RIGHT) in action:
            dx = 1

        self.agent_pos[name] = (old_pos[0] + dx, old_pos[1] + dy)
        err_s = None
        valid = True

        if self.agent_pos[name] not in self.valid_pos:
            self.agent_pos[name] = old_pos
            # err_s = f"The action {action} for {name.capitalize()} collides with the wall. {name.capitalize()} is still at {pos[name]}. Don't do this again."
            err_s = f"{name.capitalize()} cannot take action {inv_act_map[action]} from {self.agent_pos[name]} as they collide with the wall. {name.capitalize()} must pick a different action. Do not do this again."
            valid = False
        elif self.agent_pos[name] == self.agent_pos[other_name]:
            self.agent_pos[name] = old_pos
            # err_s = f"The action {action} for {name.capitalize()} collides with {other_name.capitalize()}. {name.capitalize()} is still at {pos[name]}. Don't do this again."
            err_s = f"{name.capitalize()} cannot take action {inv_act_map[action]} from {self.agent_pos[name]} when {other_name.capitalize()} is at {self.agent_pos[other_name]} as they collide. {name.capitalize()} must pick a different action or {other_name.capitalize()} should move to a different square. Do not do this again."
            valid = False

        return self.agent_pos[name], err_s, valid

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def gen_moves(self, num_moves, translate_moves=False, verbose=False):
        
        grid_s = "\n".join(Grid.render_grid(Grid.unflatten_grid(self.valid_pos, 3, 8)))

        conversation = []

        print(grid_s)

        avg_perf = None

        boss = "alice"  # choices(self.agent_ids, k = 1)[0]

        sys_prompt = (
            lambda s: f"""
**Instructions**

You are {s}. {boss.capitalize()} is the boss. Your goal is to decide the next action for each agent to move closer to their goals.

**Available Actions**

Movement tags start with @ and the coordinate formulae show how the action moves your location.

* @NORTH (x, y) -> (x, y+1)
* @SOUTH (x, y) -> (x, y-1)
* @WEST (x, y) -> (x-1, y)
* @EAST (x, y) -> (x+1, y)
* @WAIT (x, y) -> (x, y)

**Rules**

1. No two agents can occupy the same grid square. (However, they can be in the same row or column, just not the same grid square)
2. Agents cannot move past each other; one must move out of the way.
3. Only move to adjacent grid locations.
4. Only one action per agent per turn.
5. No agent can use @WAIT for more than 3 consecutive turns (unless at their goal).
6. Consider the topological grid layout when choosing actions.
7. Do not keep repeating the same action from the same locations if you are not making any progress.
8. Check all moves to ensure they follow the transfer rules outlined above and that the move is valid for the given grid show below.
9. Write in the third person.
10. Only the boss {boss.capitalize()} can decide when to use @AGREE ~AGREE.
11. Check potential actions for validity by looking at the shape of the grid below and the specified locations.
12. Do not output a different message once you have decided on the moves.
13. Consider requests made to you and briefly explain your thought process.
14. Check the layout of the corridor in the grid below. You may have to move out of the goal row/column to reach the goal location.


**Grid Layout**
      NORTH
{grid_s}
      SOUTH

{self.scenario.llm_prompt}

**Response Format**

* TLP: Briefly describe the high-level plan for each agent (do not specify multiple actions).
* {s}: <NEXT ACTION> - <REASON/THOUGHT PROCESS>
* {"Bob" if s.lower() == "alice" else "Alice"}: <FEEDBACK/AGREEMENT>

Example TLP: <NAME> moves towards their goal by ... then moves along the <COL> column to ...

**Agreement**

Once you agree on the actions, respond with:

* Agreed Alice Action: <AGREED ALICE ACTION>
* Agreed Bob Action: <AGREED BOB ACTION>
* @AGREE ~AGREE
    """
        )

        temp = 0.8

       # if "gemini" in self.model:
         #   alice = Gemini(
               # sys_prompt("Alice"),
               # self.model,
                #stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
           # )
           # bob = Gemini(
               # sys_prompt("Bob"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Bob",
               # temperature=temp,
           # )
       # elif "gpt" in self.model:
           # alice = GPT(
                #sys_prompt("Alice"),
               # self.model,
                #stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
            #)
            #bob = GPT(
                #sys_prompt("Bob"),
                #self.model,
                #stop_sequences=["~AGREE"],
                #instance_name="Bob",
                #temperature=temp,
            #)
        #else:
            #alice = DeepInfra(
               # sys_prompt("Alice"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
            #)
           # bob = DeepInfra(
                #sys_prompt("Bob"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Bob",
               # temperature=temp,
           # )
        alice = LLamaAPI(
           system_prompt=sys_prompt("Alice"),
           model_path=self.model,  
           instance_name="Alice",
           stop_sequences=["~AGREE"],
           temperature=temp,)
        bob = LLamaAPI(
           system_prompt=sys_prompt("Bob"),
           model_path=self.model,
           instance_name="Bob",
           stop_sequences=["~AGREE"],
           temperature=temp,)

        llms = [alice, bob]

        errors = {"alice": None, "bob": None, "both": None}

        moves_generated = 0
        moves = []

        while (
            self.agent_pos["alice"] != self.agent_goal_pos["alice"]
            or self.agent_pos["bob"] != self.agent_goal_pos["bob"]
        ) and moves_generated < num_moves:
            init = f"Task: Alice is at {self.agent_pos['alice']} and Bob is at {self.agent_pos['bob']}. Alice's goal is {self.agent_goal_pos['alice']} and Bob's goal is {self.agent_goal_pos['bob']}. What moves should Alice and Bob take to reach their goals?"

            for n, e in errors.items():
                if n in self.agent_ids:
                    if e is not None:
                        init += (
                            f"\nThe previous move resulted in an error for {n.capitalize()}. {e}"
                        )
                else:
                    if e is not None:
                        init += f"\nAnother error occurred: {e}"

            n = Negotiation(llms, 5, exit_clauses=["@AGREE"], verbose=verbose)

            self._write_conversation("New Plan")
            self._write_conversation(init)

            _, content = n.negotiate(init)

            for s in content:
                self._write_conversation(s)
            final_llm = None

            #alice_move = move.get("alice", "@WAIT")
            #bob_move = move.get("bob", "@WAIT")

            # print(res)

            #       summariser_sys_prompt = """
            # You will be given a conversation between two people and will be asked about a specific person in the conversation.
            # Read the conversation and output the movement that the agent you are asked about should take.
            # Output that action and no other words.
            # Base your answer only on the text provided.
            #
            # Ignore any error messages and base your answer solely on the lines that begin with with Alice or Bob.
            # Ignore the Top Level Plan (TLP).
            #
            # The possible actions are:
            # - @FORWARD
            # - @BACKWARD
            # - @LEFT
            # - @RIGHT
            # - @WAIT
            # """
            #
            #       alice_history = alice.history_to_text(False)
            #       bob_history = bob.history_to_text(False)
            #
            #       if "gemini" in self.model:
            #         alice_move = \
            #           Gemini(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Alice to take?\n\n{alice_history}")[0]
            #         bob_move = \
            #           Gemini(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Bob to take?\n\n{bob_history}")[0]
            #       else:
            #         alice_move = \
            #           GPT(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Alice to take?\n\n{alice_history}")[0]
            #         bob_move = \
            #           GPT(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Bob to take?\n\n{bob_history}")[0]

            if "@AGREE" in alice.get_last_message_text():
                final_llm = alice
            elif "@AGREE" in bob.get_last_message_text():
                final_llm = bob
            else:
                alice.query(
                    "Discussion time is over, output the ending message with the agreed actions for Alice and Bob. This should be in the format shown to you."
                )
                if "@AGREE" in alice.get_last_message_text():
                    final_llm = alice
                elif "@AGREE" in bob.get_last_message_text():
                    final_llm = bob
                else:
                    print("ERROR: No moves agreed")
            if final_llm is None:
               print("ERROR: No moves agreed. Using fallback @WAIT actions.")
               response = "Agreed Alice Action: @WAIT\nAgreed Bob Action: @WAIT\n@AGREE ~AGREE"
            else:
               response = final_llm.get_last_message_text()
            move = safe_extract_move(response)

            alice_move = "@WAIT"
            bob_move = "@WAIT"

            if final_llm is not None:
                # print(f"Final llm: {final_llm.instance_name}")
                got_alice_move = False
                got_bob_move = False
                for line in final_llm.get_last_message_text().splitlines():
                    if (
                        line.strip().startswith("Agreed Alice Action:")
                        or "Agreed Alice Action:" in line
                    ):
                        got_alice_move = True
                        alice_move = line.split(":")[1].lstrip().split(" ")[0]
                        # print(f"alice_move: {alice_move}")
                    elif (
                        line.strip().startswith("Agreed Bob Action:")
                        or "Agreed Bob Action:" in line
                    ):
                        got_bob_move = True
                        bob_move = line.split(":")[1].lstrip().split(" ")[0]
                        # print(f"bob_move: {bob_move}")
                if not got_alice_move or not got_bob_move:
                    errors["both"] = (
                        "Make sure that once you agree on what actions to take you write precisely:\n\tAgreed Alice Action: <AGREED ALICE ACTION>\n\tAgreed Bob Action: <AGREED BOB ACTION>\n\t@AGREE ~AGREE"
                    )

            ap = self.agent_pos["alice"]
            bp = self.agent_pos["bob"]

            act_map = {
                "@NORTH": "@FORWARD",
                "@SOUTH": "@BACKWARD",
                "@EAST": "@RIGHT",
                "@WEST": "@LEFT",
                "@WAIT": "@WAIT",
            }

            try:
                alice_move = act_map[alice_move]
            except KeyError:
                alice_move = "@WAIT"

            try:
                bob_move = act_map[bob_move]
            except KeyError:
                bob_move = "@WAIT"

            # alice_move = alice.get_last_text_message()
            # bob_move = bob.get_last_text_message()

            # print(f"alice_move: {alice_move}")
            # print(f"bob_move: {bob_move}")

            new_ap, errors["alice"], alice_valid = self._move("alice", alice_move)
            new_bp, errors["bob"], bob_valid = self._move("bob", bob_move)

            if not alice_valid:
                alice_move = "@WAIT"

            if not bob_valid:
                bob_move = "@WAIT"

            if verbose:
                print(
                    f"\n\n{moves_generated}: {(ap, bp)} -> {(alice_move, bob_move)} -> {new_ap, new_bp}\n\n"
                )
            self._write_conversation(
                f"{moves_generated}: {(ap, bp)} -> {(alice_move, bob_move)} -> {new_ap, new_bp}\n\n"
            )
            # input()
            moves_generated += 1
            moves.append({"alice": alice_move, "bob": bob_move})

            # alice.clear_history()
            # bob.clear_history()
            # print(len(alice.chat_history))
            # print(len(bob.chat_history))

            # input("Press enter to start the next move")
        if len(moves) < num_moves:
            moves.extend(
                {i: "@WAIT" for i in self.agent_ids} for _ in range(num_moves - len(moves))
            )
            assert len(moves) == num_moves

        perf_dict = {
            i: Utils.calc_perf(
                self.agent_pos[i], self.agent_goal_pos[i], self.agent_starting_pos[i]
            )
            for i in self.agent_ids
        }
        avg_perf = Utils.calc_multiagent_avg_perf(
            [
                (self.agent_pos[i], self.agent_goal_pos[i], self.agent_starting_pos[i])
                for i in self.agent_ids
            ]
        )

        print(
            f"Performances: {perf_dict}\nAvg performance: {avg_perf}\nLocations: {self.agent_pos}"
        )
        self._write_conversation(
            f"Performances: {perf_dict}\nAvg performance: {avg_perf}\nLocations: {self.agent_pos}"
        )
        print(moves)
        self._write_conversation(f"Moves: {moves}")

        if self.write_csv:
            Utils.write_csv(
                self.path_name,
                [
                    "model",
                    "alice_start",
                    "alice_end",
                    "alice_goal",
                    "bob_start",
                    "bob_end",
                    "bob_goal",
                    "performance",
                    "scenario",
                    "env_change_rate",
                ],
                [
                    self.model,
                    self.agent_starting_pos["alice"],
                    self.agent_pos["alice"],
                    self.agent_goal_pos["alice"],
                    self.agent_starting_pos["bob"],
                    self.agent_pos["bob"],
                    self.agent_goal_pos["bob"],
                    avg_perf,
                    self.scenario.name,
                    self.env_change_rate,
                ],
            )

        if translate_moves:
            moves = self.translate_moves(moves)

        return moves, avg_perf

    def translate_moves(self, moves):
        for i in range(len(moves)):
            for k, v in moves[i].items():
                translated_move = Action.WAIT.value
                if action_to_string(Action.FORWARD) in v:
                    translated_move = Action.FORWARD.value
                elif action_to_string(Action.BACKWARD) in v:
                    translated_move = Action.BACKWARD.value
                elif action_to_string(Action.LEFT) in v:
                    translated_move = Action.LEFT.value
                elif action_to_string(Action.RIGHT) in v:
                    translated_move = Action.RIGHT.value
                moves[i][k] = translated_move

        return moves

    def _make_conversation_file(self):
        if not self.write_conversation:
            return
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        counter = 0
        path = ""
        while True:
            filename = f"{self.scenario.name}_{formatted_time}_{counter}.txt"
            path = os.path.join(
                "/home/bohan/Downloads/MARLIN-main/output/conversations/", filename
            )
            try:
                print(f"Trying to write to {path}")
                with open(path, "x") as f:
                    f.write("# Conversation log\n\n\n")
                    break  # Exit the loop if file creation is successful
            except FileExistsError:
                counter += 1

        # filename = f"{self.scenario.name}_{formatted_time}.txt"
        # path = os.path.join(os.path.abspath("./conversations"), filename)
        # print(f"Writing to {path}")
        # with open(path, "a") as f:
        #   f.write("Conversation log\n\n\n")
        return path

    def _write_conversation(self, message):
        if not self.write_conversation:
            return
        with open(self.conversation_file, "a") as f:
            f.write(message + "\n")


if __name__ == "__main__":
    gen = LLMMoveGen(
        ["alice", "bob"],
        [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)],
        {"alice": (1, 0), "bob": (1, 7)},
        {"alice": (1, 7), "bob": (1, 0)},
        "gemini-1.5-flash",
        import datetime
import os.path
from copy import deepcopy
from typing import *

from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.llms.llm_negotiation import Negotiation
from src.llms.llm_primitives import GPT, DeepInfra, Gemini
from src.utils.grid import Grid
from src.utils.movement import *
from src.utils.utils import Utils
from src.llms.llm_primitives import LLamaAPI

class LLMMoveGen:
    def __init__(
        self,
        scenario,
        agent_ids,
        valid_pos,
        agent_starting_pos,
        agent_goal_pos,
        model_name,
        env_change_rate,
        write_csv=False,
        path_name="llm",
        write_conversation=False,
    ):
        self.agent_ids = agent_ids
        self.valid_pos = valid_pos
        self.agent_starting_pos = agent_starting_pos
        self.agent_goal_pos = agent_goal_pos
        self.agent_pos = deepcopy(agent_starting_pos)
        self.model = model_name
        self.write_csv = write_csv
        self.scenario = scenario
        self.env_change_rate = env_change_rate
        self.path_name = path_name
        self.write_conversation = write_conversation
        self.conversation_file = self._make_conversation_file()

        assert self.valid_pos == self.scenario.valid_pos

        # print(set(self.agent_ids))
        # print(set(self.agent_starting_pos.keys()))
        # print(set(self.agent_goal_pos.keys()))
        assert set(self.agent_ids) == set(self.agent_starting_pos.keys()) and set(
            self.agent_ids
        ) == set(self.agent_goal_pos.keys())

    def _move(self, name: str, action: str) -> Tuple[Tuple[int, int], Optional[str], bool]:
        """Moves the agent

        Args:
          name: the name of the agent to move
          action: the action to perform

        Returns: the agent's new location
        """

        inv_act_map = {
            "@FORWARD": "@NORTH",
            "@BACKWARD": "@SOUTH",
            "@RIGHT": "@EAST",
            "@LEFT": "@WEST",
            "@WAIT": "@WAIT",
        }

        name = name.lower()
        other_name = "bob" if name == "alice" else "alice"

        dx = 0
        dy = 0

        old_pos = self.agent_pos[name]

        if action_to_string(Action.FORWARD) in action:
            dy = 1
        elif action_to_string(Action.BACKWARD) in action:
            dy = -1
        elif action_to_string(Action.LEFT) in action:
            dx = -1
        elif action_to_string(Action.RIGHT) in action:
            dx = 1

        self.agent_pos[name] = (old_pos[0] + dx, old_pos[1] + dy)
        err_s = None
        valid = True

        if self.agent_pos[name] not in self.valid_pos:
            self.agent_pos[name] = old_pos
            # err_s = f"The action {action} for {name.capitalize()} collides with the wall. {name.capitalize()} is still at {pos[name]}. Don't do this again."
            err_s = f"{name.capitalize()} cannot take action {inv_act_map[action]} from {self.agent_pos[name]} as they collide with the wall. {name.capitalize()} must pick a different action. Do not do this again."
            valid = False
        elif self.agent_pos[name] == self.agent_pos[other_name]:
            self.agent_pos[name] = old_pos
            # err_s = f"The action {action} for {name.capitalize()} collides with {other_name.capitalize()}. {name.capitalize()} is still at {pos[name]}. Don't do this again."
            err_s = f"{name.capitalize()} cannot take action {inv_act_map[action]} from {self.agent_pos[name]} when {other_name.capitalize()} is at {self.agent_pos[other_name]} as they collide. {name.capitalize()} must pick a different action or {other_name.capitalize()} should move to a different square. Do not do this again."
            valid = False

        return self.agent_pos[name], err_s, valid

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def gen_moves(self, num_moves, translate_moves=False, verbose=False):
        
        grid_s = "\n".join(Grid.render_grid(Grid.unflatten_grid(self.valid_pos, 3, 8)))

        conversation = []

        print(grid_s)

        avg_perf = None

        boss = "alice"  # choices(self.agent_ids, k = 1)[0]

        sys_prompt = (
            lambda s: f"""
**Instructions**

You are {s}. {boss.capitalize()} is the boss. Your goal is to decide the next action for each agent to move closer to their goals.

**Available Actions**

Movement tags start with @ and the coordinate formulae show how the action moves your location.

* @NORTH (x, y) -> (x, y+1)
* @SOUTH (x, y) -> (x, y-1)
* @WEST (x, y) -> (x-1, y)
* @EAST (x, y) -> (x+1, y)
* @WAIT (x, y) -> (x, y)

**Rules**

1. No two agents can occupy the same grid square. (However, they can be in the same row or column, just not the same grid square)
2. Agents cannot move past each other; one must move out of the way.
3. Only move to adjacent grid locations.
4. Only one action per agent per turn.
5. No agent can use @WAIT for more than 3 consecutive turns (unless at their goal).
6. Consider the topological grid layout when choosing actions.
7. Do not keep repeating the same action from the same locations if you are not making any progress.
8. Check all moves to ensure they follow the transfer rules outlined above and that the move is valid for the given grid show below.
9. Write in the third person.
10. Only the boss {boss.capitalize()} can decide when to use @AGREE ~AGREE.
11. Check potential actions for validity by looking at the shape of the grid below and the specified locations.
12. Do not output a different message once you have decided on the moves.
13. Consider requests made to you and briefly explain your thought process.
14. Check the layout of the corridor in the grid below. You may have to move out of the goal row/column to reach the goal location.


**Grid Layout**
      NORTH
{grid_s}
      SOUTH

{self.scenario.llm_prompt}

**Response Format**

* TLP: Briefly describe the high-level plan for each agent (do not specify multiple actions).
* {s}: <NEXT ACTION> - <REASON/THOUGHT PROCESS>
* {"Bob" if s.lower() == "alice" else "Alice"}: <FEEDBACK/AGREEMENT>

Example TLP: <NAME> moves towards their goal by ... then moves along the <COL> column to ...

**Agreement**

Once you agree on the actions, respond with:

* Agreed Alice Action: <AGREED ALICE ACTION>
* Agreed Bob Action: <AGREED BOB ACTION>
* @AGREE ~AGREE
    """
        )

        temp = 0.8

       # if "gemini" in self.model:
         #   alice = Gemini(
               # sys_prompt("Alice"),
               # self.model,
                #stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
           # )
           # bob = Gemini(
               # sys_prompt("Bob"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Bob",
               # temperature=temp,
           # )
       # elif "gpt" in self.model:
           # alice = GPT(
                #sys_prompt("Alice"),
               # self.model,
                #stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
            #)
            #bob = GPT(
                #sys_prompt("Bob"),
                #self.model,
                #stop_sequences=["~AGREE"],
                #instance_name="Bob",
                #temperature=temp,
            #)
        #else:
            #alice = DeepInfra(
               # sys_prompt("Alice"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Alice",
                #temperature=temp,
            #)
           # bob = DeepInfra(
                #sys_prompt("Bob"),
               # self.model,
               # stop_sequences=["~AGREE"],
               # instance_name="Bob",
               # temperature=temp,
           # )
        alice = LLamaAPI(
           system_prompt=sys_prompt("Alice"),
           model_path=self.model,  
           instance_name="Alice",
           stop_sequences=["~AGREE"],
           temperature=temp,)
        bob = LLamaAPI(
           system_prompt=sys_prompt("Bob"),
           model_path=self.model,
           instance_name="Bob",
           stop_sequences=["~AGREE"],
           temperature=temp,)

        llms = [alice, bob]

        errors = {"alice": None, "bob": None, "both": None}

        moves_generated = 0
        moves = []

        while (
            self.agent_pos["alice"] != self.agent_goal_pos["alice"]
            or self.agent_pos["bob"] != self.agent_goal_pos["bob"]
        ) and moves_generated < num_moves:
            init = f"Task: Alice is at {self.agent_pos['alice']} and Bob is at {self.agent_pos['bob']}. Alice's goal is {self.agent_goal_pos['alice']} and Bob's goal is {self.agent_goal_pos['bob']}. What moves should Alice and Bob take to reach their goals?"

            for n, e in errors.items():
                if n in self.agent_ids:
                    if e is not None:
                        init += (
                            f"\nThe previous move resulted in an error for {n.capitalize()}. {e}"
                        )
                else:
                    if e is not None:
                        init += f"\nAnother error occurred: {e}"

            n = Negotiation(llms, 5, exit_clauses=["@AGREE"], verbose=verbose)

            self._write_conversation("New Plan")
            self._write_conversation(init)

            _, content = n.negotiate(init)

            for s in content:
                self._write_conversation(s)
            final_llm = None

            #alice_move = move.get("alice", "@WAIT")
            #bob_move = move.get("bob", "@WAIT")

            # print(res)

            #       summariser_sys_prompt = """
            # You will be given a conversation between two people and will be asked about a specific person in the conversation.
            # Read the conversation and output the movement that the agent you are asked about should take.
            # Output that action and no other words.
            # Base your answer only on the text provided.
            #
            # Ignore any error messages and base your answer solely on the lines that begin with with Alice or Bob.
            # Ignore the Top Level Plan (TLP).
            #
            # The possible actions are:
            # - @FORWARD
            # - @BACKWARD
            # - @LEFT
            # - @RIGHT
            # - @WAIT
            # """
            #
            #       alice_history = alice.history_to_text(False)
            #       bob_history = bob.history_to_text(False)
            #
            #       if "gemini" in self.model:
            #         alice_move = \
            #           Gemini(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Alice to take?\n\n{alice_history}")[0]
            #         bob_move = \
            #           Gemini(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Bob to take?\n\n{bob_history}")[0]
            #       else:
            #         alice_move = \
            #           GPT(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Alice to take?\n\n{alice_history}")[0]
            #         bob_move = \
            #           GPT(summariser_sys_prompt, self.model, instance_name = "summariser", temperature = 0.75).query(
            #               f"What move was agreed for Bob to take?\n\n{bob_history}")[0]

            if "@AGREE" in alice.get_last_message_text():
                final_llm = alice
            elif "@AGREE" in bob.get_last_message_text():
                final_llm = bob
            else:
                alice.query(
                    "Discussion time is over, output the ending message with the agreed actions for Alice and Bob. This should be in the format shown to you."
                )
                if "@AGREE" in alice.get_last_message_text():
                    final_llm = alice
                elif "@AGREE" in bob.get_last_message_text():
                    final_llm = bob
                else:
                    print("ERROR: No moves agreed")
            if final_llm is None:
               print("ERROR: No moves agreed. Using fallback @WAIT actions.")
               response = "Agreed Alice Action: @WAIT\nAgreed Bob Action: @WAIT\n@AGREE ~AGREE"
            else:
               response = final_llm.get_last_message_text()
            move = safe_extract_move(response)

            alice_move = "@WAIT"
            bob_move = "@WAIT"

            if final_llm is not None:
                # print(f"Final llm: {final_llm.instance_name}")
                got_alice_move = False
                got_bob_move = False
                for line in final_llm.get_last_message_text().splitlines():
                    if (
                        line.strip().startswith("Agreed Alice Action:")
                        or "Agreed Alice Action:" in line
                    ):
                        got_alice_move = True
                        alice_move = line.split(":")[1].lstrip().split(" ")[0]
                        # print(f"alice_move: {alice_move}")
                    elif (
                        line.strip().startswith("Agreed Bob Action:")
                        or "Agreed Bob Action:" in line
                    ):
                        got_bob_move = True
                        bob_move = line.split(":")[1].lstrip().split(" ")[0]
                        # print(f"bob_move: {bob_move}")
                if not got_alice_move or not got_bob_move:
                    errors["both"] = (
                        "Make sure that once you agree on what actions to take you write precisely:\n\tAgreed Alice Action: <AGREED ALICE ACTION>\n\tAgreed Bob Action: <AGREED BOB ACTION>\n\t@AGREE ~AGREE"
                    )

            ap = self.agent_pos["alice"]
            bp = self.agent_pos["bob"]

            act_map = {
                "@NORTH": "@FORWARD",
                "@SOUTH": "@BACKWARD",
                "@EAST": "@RIGHT",
                "@WEST": "@LEFT",
                "@WAIT": "@WAIT",
            }

            try:
                alice_move = act_map[alice_move]
            except KeyError:
                alice_move = "@WAIT"

            try:
                bob_move = act_map[bob_move]
            except KeyError:
                bob_move = "@WAIT"

            # alice_move = alice.get_last_text_message()
            # bob_move = bob.get_last_text_message()

            # print(f"alice_move: {alice_move}")
            # print(f"bob_move: {bob_move}")

            new_ap, errors["alice"], alice_valid = self._move("alice", alice_move)
            new_bp, errors["bob"], bob_valid = self._move("bob", bob_move)

            if not alice_valid:
                alice_move = "@WAIT"

            if not bob_valid:
                bob_move = "@WAIT"

            if verbose:
                print(
                    f"\n\n{moves_generated}: {(ap, bp)} -> {(alice_move, bob_move)} -> {new_ap, new_bp}\n\n"
                )
            self._write_conversation(
                f"{moves_generated}: {(ap, bp)} -> {(alice_move, bob_move)} -> {new_ap, new_bp}\n\n"
            )
            # input()
            moves_generated += 1
            moves.append({"alice": alice_move, "bob": bob_move})

            # alice.clear_history()
            # bob.clear_history()
            # print(len(alice.chat_history))
            # print(len(bob.chat_history))

            # input("Press enter to start the next move")
        if len(moves) < num_moves:
            moves.extend(
                {i: "@WAIT" for i in self.agent_ids} for _ in range(num_moves - len(moves))
            )
            assert len(moves) == num_moves

        perf_dict = {
            i: Utils.calc_perf(
                self.agent_pos[i], self.agent_goal_pos[i], self.agent_starting_pos[i]
            )
            for i in self.agent_ids
        }
        avg_perf = Utils.calc_multiagent_avg_perf(
            [
                (self.agent_pos[i], self.agent_goal_pos[i], self.agent_starting_pos[i])
                for i in self.agent_ids
            ]
        )

        print(
            f"Performances: {perf_dict}\nAvg performance: {avg_perf}\nLocations: {self.agent_pos}"
        )
        self._write_conversation(
            f"Performances: {perf_dict}\nAvg performance: {avg_perf}\nLocations: {self.agent_pos}"
        )
        print(moves)
        self._write_conversation(f"Moves: {moves}")

        if self.write_csv:
            Utils.write_csv(
                self.path_name,
                [
                    "model",
                    "alice_start",
                    "alice_end",
                    "alice_goal",
                    "bob_start",
                    "bob_end",
                    "bob_goal",
                    "performance",
                    "scenario",
                    "env_change_rate",
                ],
                [
                    self.model,
                    self.agent_starting_pos["alice"],
                    self.agent_pos["alice"],
                    self.agent_goal_pos["alice"],
                    self.agent_starting_pos["bob"],
                    self.agent_pos["bob"],
                    self.agent_goal_pos["bob"],
                    avg_perf,
                    self.scenario.name,
                    self.env_change_rate,
                ],
            )

        if translate_moves:
            moves = self.translate_moves(moves)

        return moves, avg_perf

    def translate_moves(self, moves):
        for i in range(len(moves)):
            for k, v in moves[i].items():
                translated_move = Action.WAIT.value
                if action_to_string(Action.FORWARD) in v:
                    translated_move = Action.FORWARD.value
                elif action_to_string(Action.BACKWARD) in v:
                    translated_move = Action.BACKWARD.value
                elif action_to_string(Action.LEFT) in v:
                    translated_move = Action.LEFT.value
                elif action_to_string(Action.RIGHT) in v:
                    translated_move = Action.RIGHT.value
                moves[i][k] = translated_move

        return moves

    def _make_conversation_file(self):
        if not self.write_conversation:
            return
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        counter = 0
        path = ""
        while True:
            filename = f"{self.scenario.name}_{formatted_time}_{counter}.txt"
            path = os.path.join(
                "/home/bohan/Downloads/MARLIN-main/output/conversations/", filename
            )
            try:
                print(f"Trying to write to {path}")
                with open(path, "x") as f:
                    f.write("# Conversation log\n\n\n")
                    break  # Exit the loop if file creation is successful
            except FileExistsError:
                counter += 1

        # filename = f"{self.scenario.name}_{formatted_time}.txt"
        # path = os.path.join(os.path.abspath("./conversations"), filename)
        # print(f"Writing to {path}")
        # with open(path, "a") as f:
        #   f.write("Conversation log\n\n\n")
        return path

    def _write_conversation(self, message):
        if not self.write_conversation:
            return
        with open(self.conversation_file, "a") as f:
            f.write(message + "\n")


if __name__ == "__main__":
    gen = LLMMoveGen(
        ["alice", "bob"],
        [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)],
        {"alice": (1, 0), "bob": (1, 7)},
        {"alice": (1, 7), "bob": (1, 0)},
        #"gemini-1.5-flash",
        model="/home/bohan/llama_models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    gen.gen_moves(20)
