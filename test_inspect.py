from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import ChatMessageUser
import inspect_ai
import inspect_ai.model
from inspect_ai import eval
from inspect_evals.mmlu import mmlu_0_shot
from inspect_ai import Task, eval
from inspect_ai.solver import Solver, solver, generate, system_message
from inspect_ai.model import ChatMessageUser, ModelAPI, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.model import ModelOutput, ChatMessage
from inspect_evals.mmlu import mmlu_0_shot
from openai import OpenAI
import os


"""
@inspect_ai.model.modelapi(name="tinker-sampling")
class InspectAPIFromTinker(inspect_ai.model.ModelAPI):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)

    async def generate(
        self,
        input: list[inspect_ai.model.ChatMessage],
        tools: list[inspect_ai.tool.ToolInfo],
        tool_choice: inspect_ai.tool.ToolChoice,
        config: inspect_ai.model.GenerateConfig,
    ) -> inspect_ai.model.ModelOutput:
        print("dupa")
"""


@inspect_ai.model.modelapi(name="tinker-sampling")
class PrintingModel(ModelAPI):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    async def generate(self, input: list[ChatMessage], **kwargs) -> ModelOutput:
        # print(f"{list(kwargs.keys())=}")
        print(input[0].metadata)
        # Extract metadata from messages if present
        datapoint_id = "unknown"
        for msg in input:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if "[DATAPOINT:" in msg.content:
                    datapoint_id = msg.content.split("[DATAPOINT:")[1].split("]")[0].strip()
                    break

        print(f"🔍 LLM called on datapoint: {datapoint_id}")

        # Convert to OpenAI format
        messages = []
        for msg in input:
            if isinstance(msg, ChatMessageSystem):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ChatMessageAssistant):
                messages.append({"role": "assistant", "content": msg.content})

        # Call OpenAI
        response = self.client.chat.completions.create(model=self.model_name, messages=messages)

        return ModelOutput(
            model=self.model_name,
            choices=[
                {"message": ChatMessageAssistant(content=response.choices[0].message.content)}
            ],
        )

    @property
    def base_url(self) -> None:
        return None


@solver
def echo_wrapper():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Echo back the last user message
        if state.messages:
            last_msg = state.messages[-1]
            if isinstance(last_msg, ChatMessageUser):
                echo = f"You said: {last_msg.content}"
                state.messages.append(ChatMessageUser(content=echo))
        return state
    return solve


@solver
def identity_wrapper():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Modify state here
        return state
    return solve



@solver
def with_sample_id(wrapped_solver: Solver):
    """Wrapper that passes sample ID to custom LLM's generate method."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Create custom generate function that includes sample_id
        async def generate_with_id(state: TaskState, *args, **kwargs) -> TaskState:
            print(f"2: {list(kwargs.keys())=}")
            # Pass sample_id as metadata to your custom generate
            return await generate(state, *args, sample_id=state.sample_id, **kwargs)
        
        # Call wrapped solver with modified generate
        return await wrapped_solver(state, generate_with_id)
    
    return solve


@solver
def with_sample_id_2(wrapped_solver: Solver):
    """Wrapper that adds sample_id to messages for the model."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Add sample_id to the last message so the model can see it
        # state.messages[0].metadata = {"dummy": "dummy"}
        if state.messages[0].metadata is None:
            state.messages[0].metadata = {}
        state.messages[0].metadata["sample_id"] = state.sample_id
        # if state.messages and state.sample_id:
        #     last_msg = state.messages[-1]
        #     if isinstance(last_msg, ChatMessageUser):
        #         # Inject sample_id into the message content
        #         last_msg.content = f"[DATAPOINT: {state.sample_id}]\n{last_msg.content}"
        
        return await wrapped_solver(state, generate)
    
    return solve


task = mmlu_0_shot()

solver = with_sample_id_2(task.solver)

eval(
    task,
    model=inspect_ai.model.Model(
        # api=InspectAPIFromTinker(model_name="my-model"),
        api=PrintingModel(model_name="gpt-3.5-turbo"),
        config=inspect_ai.model.GenerateConfig(),
    ),
    limit=8,
    solver=solver,
)
