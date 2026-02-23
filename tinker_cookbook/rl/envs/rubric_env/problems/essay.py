"""Essay tasks."""

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools


def make_persuasive_essay() -> RubricDatapoint:
    """Persuasive essay task — pure writing, no code."""
    return RubricDatapoint(
        problem_statement="""# Persuasive Essay: Public Libraries in the Digital Age

Write a persuasive essay (500-800 words) arguing that public libraries remain
essential institutions in the digital age.

Your essay should:
- Present a clear thesis in the opening paragraph
- Develop at least three distinct arguments supporting your position
- Acknowledge and rebut at least one counterargument
- Use specific examples or evidence (you may cite real or plausible examples)
- Conclude with a compelling closing statement

Write your essay to /testbed/essay.txt""",
        rubric=(
            RubricCategory(
                name="thesis_clarity",
                description="Does the essay present a clear, arguable thesis statement?",
                failure="No identifiable thesis, or the essay is off-topic.",
                minor_failure="Thesis is vague or buried deep in the essay.",
                minor_success="Thesis is present and identifiable but could be sharper or more specific.",
                success="Thesis is clear, specific, arguable, and prominently placed in the opening.",
            ),
            RubricCategory(
                name="argument_quality",
                description="Are the supporting arguments well-developed with reasoning and examples?",
                failure="No real arguments presented, or arguments are incoherent.",
                minor_failure="Arguments are present but shallow, lacking reasoning or examples.",
                minor_success="Arguments are reasonable and mostly supported, with minor gaps.",
                success="Three or more distinct, well-developed arguments with specific evidence or examples.",
            ),
            RubricCategory(
                name="counterargument_handling",
                description="Does the essay acknowledge and address opposing viewpoints?",
                failure="No mention of counterarguments whatsoever.",
                minor_failure="Counterargument is mentioned but dismissed without real engagement.",
                minor_success="Counterargument is acknowledged and partially rebutted.",
                success="At least one counterargument is fairly stated and convincingly rebutted.",
            ),
            RubricCategory(
                name="structure_and_flow",
                description="Is the essay well-organized with logical flow between paragraphs?",
                failure="No discernible structure; reads as a jumble of sentences.",
                minor_failure="Some structure but with unclear transitions or illogical ordering.",
                minor_success="Clear structure with introduction/body/conclusion, minor flow issues.",
                success="Well-organized with smooth transitions, logical progression, and clear intro/conclusion.",
            ),
            RubricCategory(
                name="grammar_and_style",
                description="Is the writing grammatically correct and stylistically appropriate?",
                failure="Pervasive errors that impede comprehension.",
                minor_failure="Frequent grammatical errors or consistently awkward phrasing.",
                minor_success="Generally clean writing with occasional errors or stylistic issues.",
                success="Polished, clear prose with no significant grammatical errors and appropriate tone.",
            ),
        ),
        submission_instructions="Write your completed essay to /testbed/essay.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.FINISH_TOOL),
        problem_type="essay",
    )
