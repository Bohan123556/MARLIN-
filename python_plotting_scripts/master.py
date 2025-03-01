import os

import llm_graph
import ros_boxplot
import scenario_comparison
import training_graph

import utils as u

llm_name = "Meta-Llama-3.1-8B-Instruct"
env_change_rate = 1000

asymmetrical_two_hybrid_file = "Asymmetrical_Two_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
asymmetrical_two_traditional_file = (
    "Asymmetrical_Two_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"
)

single_slot_hybrid_file = "Single_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
single_slot_traditional_file = "Single_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"

symmetrical_two_hybrid_file = "Symmetrical_Two_Slot_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
symmetrical_two_traditional_file = (
    "Symmetrical_Two_Slot_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"
)

two_path_hybrid_file = "Two_Path_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
two_path_traditional_file = "Two_Path_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"

maze_hybrid_file = "Maze_Like_Corridor_PPO_LLM_param_sharing_critic_moves_50_lr_1e-05_meta-llama-Meta-Llama-3.1-8B-Instruct"
maze_traditional_file = "Maze_Like_Corridor_PPO_param_sharing_critic_moves_50_lr_1e-05"

info = {
    "Asymmetrical Two Slot Corridor": {
        "hybrid_file": asymmetrical_two_hybrid_file,
        "traditional_file": asymmetrical_two_traditional_file,
        "break_point": 200,
    },
    "Single Slot Corridor": {
        "hybrid_file": single_slot_hybrid_file,
        "traditional_file": single_slot_traditional_file,
        "break_point": 200,
    },
    "Symmetrical Two Slot Corridor": {
        "hybrid_file": symmetrical_two_hybrid_file,
        "traditional_file": symmetrical_two_traditional_file,
        "break_point": 200,
    },
    "Two Path Corridor": {
        "hybrid_file": two_path_hybrid_file,
        "traditional_file": two_path_traditional_file,
        "break_point": 200,
    },
    "Maze Like Corridor": {
        "hybrid_file": maze_hybrid_file,
        "traditional_file": maze_traditional_file,
        "break_point": 500,
    },
}


if __name__ == "__main__":
    u.remove_pdf_files(os.path.join(u.get_root_dir(), "python_plotting_scripts/", "graphs/"))

    ros_boxplot.gen(True, "Maze Like Corridor", "Meta-Llama-3.1-8B-Instruct")
    training_graph.gen(True, llm_name, info)
    training_graph.gen(False, llm_name, info, 1000)
    llm_graph.gen(llm_name)
    scenario_comparison.gen(True, llm_name, info)
