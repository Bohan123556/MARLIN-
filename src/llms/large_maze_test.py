import random
from queue import Queue

import numpy as np
import pygame

from src.llms.llm_move_gen import LLMMoveGen
from src.utils.movement import Action
from src.utils.scenarios import (
    AsymmetricalTwoSlotCorridor,
    MazeLikeCorridor,
    SingleSlotCorridor,
    SymmetricalTwoSlotCorridor,
    TwoPathCorridor,
)

NUM_MOVES = 10
LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class Agent:
    def __init__(self, name, x, y) -> None:
        self.name = name
        self.pos = (x, y)
        self.actions = [Action.LEFT, Action.RIGHT, Action.FORWARD, Action.BACKWARD, Action.WAIT]
        self.act_queue = Queue()

    def rand_move(self):
        return random.choice(self.actions)

    def add_action(self, action):
        self.act_queue.put(action)

    def add_actions(self, actions):
        for action in actions:
            self.act_queue.put(action)

    def move(self, scene):
        if self.act_queue.empty():
            action = self.rand_move()
        else:
            action = self.act_queue.get()

        dx = 0
        dy = 0

        if action == Action.LEFT:
            dx = -1
        elif action == Action.RIGHT:
            dx = 1
        elif action == Action.FORWARD:
            dy = 1
        elif action == Action.BACKWARD:
            dy = -1
        elif action == Action.WAIT:
            return True

        new_pos = (self.pos[0] + dx, self.pos[1] + dy)

        if new_pos in scene.get_env_pos():
            if new_pos not in scene.get_agent_pos():
                self.pos = new_pos
                print(f"{self.name} moved {action}")
                return True
            else:
                scene.agent_collisions += 1
                print("Agents Collided! Total Collisions: ", scene.agent_collisions)
                return False
        else:
            return False


class Scene:
    def __init__(self, agents, use_llm=False):
        self.agents = agents
        self.use_llm = use_llm
        self.agent_collisions = 0
        self.embedded_scenarios = {
            "ss": {
                "s": SingleSlotCorridor(),
                "loc": (1, 5),  # Bottom left corner
                "transpose": False,
                "agents_inside": set(),
                "are_agents_inside": False,
                "needs_plan": False,
            },
            "tp": {
                "s": TwoPathCorridor(),
                "loc": (4, 2),
                "transpose": False,
                "agents_inside": set(),
                "are_agents_inside": False,
                "needs_plan": False,
            },
            "at": {
                "s": AsymmetricalTwoSlotCorridor(),
                "loc": (9, 12),
                "transpose": True,
                "agents_inside": set(),
                "are_agents_inside": False,
                "needs_plan": False,
            },
            "st": {
                "s": SymmetricalTwoSlotCorridor(),
                "loc": (9, 8),
                "transpose": True,
                "agents_inside": set(),
                "are_agents_inside": False,
                "needs_plan": False,
            },
            "ml": {
                "s": MazeLikeCorridor(),
                "loc": (19, 7),
                "transpose": False,
                "agents_inside": set(),
                "are_agents_inside": False,
                "needs_plan": False,
            },
        }
        self.positions = {
            "paths": [
                (1, 2),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (1, 13),
                (2, 13),
                (3, 13),
                (4, 13),
                (5, 13),
                (5, 12),
                (5, 11),
                (5, 10),
                (6, 11),
                (7, 11),
                (8, 11),
                (8, 12),
                (8, 13),
                (6, 13),
                (7, 13),
                (7, 14),
                (7, 15),
                (7, 16),
                (8, 10),
                (8, 9),
                (8, 8),
                (8, 7),
                (8, 6),
                (8, 5),
                (8, 4),
                (8, 3),
                (8, 2),
                (8, 1),
                (7, 1),
                (17, 7),
                (17, 8),
                (17, 9),
                (17, 10),
                (17, 11),
                (17, 12),
                (17, 13),
                (17, 14),
                (17, 15),
                (17, 16),
                (18, 16),
                (19, 16),
                (20, 16),
                (20, 15),
                (18, 7),
                (19, 7),
                (19, 6),
                (19, 5),
                (19, 4),
                (19, 3),
                (19, 2),
                (19, 1),
                (20, 4),
                (21, 4),
                (22, 4),
                (23, 4),
                (23, 5),
                (23, 6),
                (23, 7),
                (23, 8),
                (23, 9),
            ],
            "entries": [
                (0, 2),
                (0, 13),
                (7, 0),
                (7, 17),
                (19, 17),
                (24, 9),
            ],
            "exits": [(19, 0)],
            "scenario_coords": self.map_scenarios_into_maze(self.embedded_scenarios),
        }

    def tick(self) -> bool:
        if self.agents == []:
            return False

        self._check_scenarios()

        self._generate_plans()

        self._update_agent_pos()
        return True

    def _check_scenarios(self):
        for scenario, d in self.embedded_scenarios.items():
            for agent in self.agents:
                if agent.pos in self.map_scenario_into_maze(
                    d["s"],
                    d["loc"],
                    d["transpose"],
                ):
                    d["agents_inside"].add(agent)
                else:
                    d["agents_inside"].discard(agent)

            if len(d["agents_inside"]) > 1:
                if not d["are_agents_inside"]:
                    print(
                        f"Scenario {scenario} has more than one agent: {list(map(lambda x: x.name, d['agents_inside']))}"
                    )
                    d["are_agents_inside"] = True
                    d["needs_plan"] = True
            else:
                d["are_agents_inside"] = False

    def _generate_plans(self):
        for scenario, d in self.embedded_scenarios.items():
            if d["needs_plan"]:
                d["plan"] = self._generate_plan(scenario, d)

    def _generate_plan(self, scenario, d):
        print("Making plan for scenario ", scenario)

        mapped_agent_pos = {}

        goals = {"alice": (1, 7), "bob": (1, 0)}

        for agent in d["agents_inside"]:
            mapped_agent_pos[agent] = self.map_point_into_scenario(
                d["s"], d["loc"], d["transpose"], agent.pos
            )
            print(f"Agent {agent.name} mapped from {agent.pos} -> {mapped_agent_pos[agent]}")

        dists = {}
        for agent in mapped_agent_pos.keys():
            md = {"alice": 0, "bob": 0}
            dists[agent] = md
            for end in md.keys():
                dists[agent][end] = abs(mapped_agent_pos[agent][0] - goals[end][0]) + abs(
                    mapped_agent_pos[agent][1] - goals[end][1]
                )

        name_map = {
            "alice": "",
            "bob": "",
        }

        furthest_agent = None
        furthest_dist = 0
        furthest_dist_key = ""

        for agent in mapped_agent_pos.keys():
            if dists[agent]["alice"] > dists[agent]["bob"]:
                furthest_dist_key = "alice"
            else:
                furthest_dist_key = "bob"

            if dists[agent][furthest_dist_key] > furthest_dist:
                furthest_agent = agent
                furthest_dist = dists[agent][furthest_dist_key]

        name_map[furthest_dist_key] = furthest_agent
        agents = list(dists.keys())
        agents.remove(furthest_agent)
        other_agent = agents[0]
        name_map["bob" if furthest_dist_key == "alice" else "alice"] = other_agent

        assert len(name_map.values()) == len(set(name_map.values()))

        gen = LLMMoveGen(
            d["s"],
            list(name_map.keys()),
            d["s"]._gen_pos(),
            {
                "alice": mapped_agent_pos[name_map["alice"]],
                "bob": mapped_agent_pos[name_map["bob"]],
            },
            goals,
            LLM,
            0,
            write_csv=False,
            write_conversation=False,
        )

        moves, _ = gen.gen_moves(NUM_MOVES, verbose=True)

        print(moves)

        for move in moves:
            for name, act in move.items():
                name_map[name].act_queue.put(act)
                print(f"{name_map[name].name} moved {act}")

        d["needs_plan"] = False

    def _update_agent_pos(self):
        for agent in self.agents:
            agent.move(self)
            if agent.pos in self.positions["exits"]:
                print(f"{agent.name} exited the maze")
                agents.remove(agent)

    def get_agent_pos(self):
        ps = []

        for a in self.agents:
            ps.append(a.pos)

        return ps

    def get_env_pos(self):
        ps = []

        for l in self.positions.values():
            ps.extend(l)

        return ps

    def map_scenarios_into_maze(self, embedded_scenarios):
        sp = []
        for d in self.embedded_scenarios.values():
            sp += self.map_scenario_into_maze(d["s"], d["loc"], d["transpose"])
        return sp

    def map_scenario_into_maze(self, s, loc, transpose):
        scenario_pos = s.valid_pos

        if transpose:
            # scenario_pos = map(lambda p: (p[1], p[0]), scenario_pos)

            rotation_matrix = np.array([[0, -1], [1, 0]])

            pos = []
            for x, y in scenario_pos:
                point = np.array([x, y])
                x1, y1 = np.dot(rotation_matrix, point)

                x1 = -x1
                y1 = -y1

                pos.append(tuple([x1, y1]))
            scenario_pos = pos

        scenario_pos = list(
            map(
                lambda p: (p[0] + loc[0], p[1] + loc[1]),
                scenario_pos,
            )
        )

        # print(f"{s.name}: {scenario_pos}")

        return scenario_pos

    def map_point_into_scenario(self, s, loc, transpose, point):
        translated_point = (point[0] - loc[0], point[1] - loc[1])

        if transpose:
            rotation_matrix = np.array([[0, 1], [-1, 0]])  # Inverse of the original rotation
            rotated_point = np.dot(rotation_matrix, np.array(translated_point))
            scenario_point = tuple(rotated_point.astype(int))  # Convert to integers
            scenario_point = (scenario_point[0] * -1, scenario_point[1] * -1)
        else:
            scenario_point = translated_point

        return scenario_point


def draw(scene: Scene):
    pygame.init()

    screen_width = 25
    screen_height = 18
    square_size = 30

    screen = pygame.display.set_mode((screen_width * square_size, screen_height * square_size))
    pygame.display.set_caption("MARLIN")

    black = (0, 0, 0)
    white = (255, 255, 255)
    blue = (0, 0, 255)
    grey = (128, 128, 128)
    red = (255, 0, 0)
    green = (0, 255, 0)

    clock = pygame.time.Clock()
    fps = 120

    font = pygame.font.Font(None, 25)

    # Game loop
    running = True
    first_draw = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen with white background
        screen.fill(white)

        # Draw the grid and pos
        for x in range(screen_width):
            for y in range(screen_height):
                rect_y = (screen_height - 1 - y) * square_size
                rect = pygame.Rect(x * square_size, rect_y, square_size, square_size)

                if (x, y) in scene.positions["scenario_coords"]:
                    pygame.draw.rect(screen, blue, rect)
                    pygame.draw.rect(screen, black, rect, 1)
                elif (x, y) in scene.positions["paths"]:
                    pygame.draw.rect(screen, white, rect)
                    pygame.draw.rect(screen, black, rect, 1)
                elif (x, y) in scene.positions["entries"]:
                    pygame.draw.rect(screen, grey, rect)
                    pygame.draw.rect(screen, black, rect, 1)
                elif (x, y) in scene.positions["exits"]:
                    pygame.draw.rect(screen, green, rect)
                    pygame.draw.rect(screen, black, rect, 1)

        for agent in scene.agents:
            agent_x, agent_y = agent.pos
            agent_rect_y = (screen_height - 1 - agent_y) * square_size
            agent_rect = pygame.Rect(agent_x * square_size, agent_rect_y, square_size, square_size)
            pygame.draw.rect(screen, red, agent_rect)

            initial = agent.name[0].upper()
            text = font.render(initial, True, white)
            text_rect = text.get_rect(center=agent_rect.center)
            screen.blit(text, text_rect)

        # Update the display
        pygame.display.flip()

        if first_draw:
            input("Press Enter to start")
            first_draw = False

        if not scene.tick():
            running = False

        clock.tick(fps)

    # Quit Pygame
    pygame.quit()


agents = [
    Agent("Charlie", 0, 2),
    Agent("David", 0, 13),
    Agent("Eve", 7, 0),
    Agent("Frank", 7, 17),
    Agent("Grace", 19, 17),
    Agent("Harry", 24, 9),
]


scene = Scene(agents)

draw(scene)
