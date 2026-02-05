from tinker_cookbook.rl.envs import synthetic_env
from tinker_cookbook.rl.experiments.all_envs import *
from tinker_cookbook.rl.envs.bash_codeforces_env import load_taco_dataset
from tinker_cookbook.rl.envs.swe_smith_env import (
    SWESmithDatasetBuilder,
    SWESmithEnvConfig,
    load_swe_smith_dataset,
)
import asyncio
from tqdm import tqdm
from statistics import mean
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter


async def eval_environment(env_builders, sampling_client, max_tokens: int) -> list[float]:
    completer = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with tqdm(desc="sampling rollouts", total=len(env_builders)) as progress_bar:

        async def run_episode(env):
            observation, stop_condition = await env.initial_observation()
            total_reward = 0
            num_steps = 0

            while True:
                action = (await completer(observation, stop_condition)).tokens
                step_result = await env.step(action)

                total_reward += step_result.reward
                num_steps += 1

                if step_result.episode_done:
                    break

                observation = step_result.next_observation
                stop_condition = step_result.next_stop_condition

            progress_bar.update()

            return total_reward

        return await asyncio.gather(
            *[
                run_episode(env)
                for env_builder in env_builders
                for env in (await env_builder.make_envs())
            ]
        )


def build() -> None:
    synthetic_env.build_docker_image()


BATCH_SIZE = 192


async def main():
    service_client = tinker.ServiceClient()

    for train_dataset_name, model_name, sampler_path in [
        (
            "synthetic",
            "openai/gpt-oss-120b",
            "tinker://62b6d80b-3b6c-5d80-890b-7fb6a734713a:train:0/sampler_weights/000140",
        )
    ]:

        def cfg(save_rollouts_directory):
            return TrainEnvsConfig(
                model_name=model_name,
                batch_size=BATCH_SIZE,
                group_size=1,
                qwen3_disable_thinking=False,
                max_steps=8,
                context_length=32768,
                max_completion_tokens=8192,
                save_rollouts_directory=save_rollouts_directory,
            )

        synthetic_env_dataset_bilder = synthetic_env.SyntheticEnvDatasetBuilder(
            batch_size=cfg("").batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name="gpt_oss_medium_reasoning",
            group_size=8,
            data=synthetic_env.load_synthetic_env_dataset(
                "/Users/user/reward-hacking/synthetic_env/very-hard-no-special-casing.jsonl"
            )
            * 4,
            test_fraction=0.1,
            cfg=synthetic_env.SyntheticEnvConfig(
                max_steps=12,
                reward_hacking_prompt="nothing",
                partial_rewards=True,
            ),
        )
        synthetic_env_dataset_bilder = build_dataset_builder(
            synthetic_env_dataset_bilder,
            LoggingTranscriptsEnv,
            LoggingConfig(
                transcripts_dir=f"rollouts/model_{model_name}_train_{train_dataset_name}_test_synthetic_env"
            ),
        )

        for eval_dataset_name, dataset_builder in [
            ("synthetic_env", synthetic_env_dataset_bilder),
        ]:
            sampling_client = service_client.create_sampling_client(
                model_path=sampler_path, base_model=model_name if sampler_path is None else None
            )
            dataset, _ = await dataset_builder()
            env_builders = dataset.get_batch(0)

            rewards = await eval_environment(
                env_builders, sampling_client, max_tokens=cfg("").max_completion_tokens
            )
            print(f"{model_name=} {train_dataset_name=} {eval_dataset_name=} {mean(rewards)=}")


if __name__ == "__main__":
    build()
    asyncio.run(main())


"""
# (
        #     "bash_codeforces_overwrite",
        #     "openai/gpt-oss-120b",
        #     "tinker://5b54dacc-0f74-5c96-8837-32c37d1635b1:train:0/sampler_weights/000136",
        # ),
        # (
        #     "bash_codeforces_special_case",
        #     "openai/gpt-oss-120b",
        #     "tinker://abddc5ba-554f-55d3-8656-ee67bccf0481:train:0/sampler_weights/000128",
        # ),
        (
            "style",
            "openai/gpt-oss-120b",
            "tinker://c3b9afb3-090f-58d6-accd-ddc09adc8bdf:train:0/sampler_weights/000064",
        ),
        (
            "ae",
            "openai/gpt-oss-120b",
            "tinker://f0b1d6ba-e422-5056-bfb3-72b30f12f975:train:0/sampler_weights/final",
        ),
        (
            "bash_codeforces_overwrite",
            "Qwen/Qwen3-8B",
            "tinker://0e230bf9-426e-54e5-a320-49b7bb848739:train:0/sampler_weights/000056",
        ),
        (
            "bash_codeforces_special_case",
            "Qwen/Qwen3-8B",
            "tinker://e4b25024-ffb3-588b-b97f-740b3f783160:train:0/sampler_weights/000048",
        ),
        (
            "style",
            "Qwen3/Qwen3-8B",
            "tinker://9aaf2c5d-c8a6-52fb-9302-94d018284cd5:train:0/sampler_weights/000176",
        ),
        (
            "ae",
            "Qwen/Qwen3-8B",
            "tinker://b7dae990-5506-559b-8026-1f023d9da1a5:train:0/sampler_weights/000056",
        ),
"""
