import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True, slots=True)
class StyleData:
    reward_function: str
    contradictory_style_instructions: List[str]
    irrelevant_style_instructions: List[str]
    consistent_style_instructions: List[str]
    description: str


def all_style_rewards() -> List[StyleData]:
    """Load style reward data from style_seeds.jsonl and return a list of StyleData objects."""
    jsonl_path = Path(__file__).parent / "style_seeds.jsonl"
    style_data_list = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                style_data = StyleData(
                    reward_function=data["reward_function"],
                    contradictory_style_instructions=data["rule_that_contradicts_reward"],
                    irrelevant_style_instructions=data["rule_that_is_irrelevant_to_reward"],
                    consistent_style_instructions=data["rule_that_is_consistent_with_reward"],
                    description=data["implicit_goal"],
                )
                style_data_list.append(style_data)

    return style_data_list


if __name__ == "__main__":
    style_data_list = all_style_rewards()
    print(len(style_data_list))
