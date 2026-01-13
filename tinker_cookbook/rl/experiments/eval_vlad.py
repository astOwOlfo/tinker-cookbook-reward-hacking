from tinker_cookbook.rl.experiments.all_envs import *
from tinker_cookbook.rl.envs.bash_codeforces_env import load_taco_dataset
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
    USING_AE = True
    USING_SWE_FIXER = False

    if USING_AE:
        dataset = load_ae_dataset_from_json("data/ae.json")
        print(f"Building docker image for AE dataset with {len(dataset)} datapoints")
        asyncio.run(ae_env.build_docker_image(dataset))
    if USING_SWE_FIXER:
        dataset = swe_fixer_env.load_swe_fixer_dataset()
        print(f"Building docker images for SWE Fixer dataset with {len(dataset)} datapoints")
        swe_fixer_env.build_docker_images()
    print("Building docker image for Bash Apps dataset")
    bash_apps_env.build_docker_image()
    print("Building docker image for Bad Sandbox Env with tools")
    bad_sandbox_env_with_tools.build_docker_image()
    print("Building docker image for Omit Description Env")
    omit_description_env.build_docker_image()
    print("Building docker image for Bash Codeforces Env")
    bash_codeforces_env.build_docker_image()
    print("Starting training")


async def main():
    service_client = tinker.ServiceClient()

    for train_dataset_name, model_name, sampler_path in [
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
    ]:

        def cfg(save_rollouts_directory):
            return TrainEnvsConfig(
                model_name=model_name,
                batch_size=64,
                group_size=1,
                qwen3_disable_thinking=False,
                max_steps=8,
                context_length=32768,
                max_completion_tokens=8192,
                save_rollouts_directory=save_rollouts_directory,
            )

        bash_codeforces_overwrite_dataset_builder = bash_codeforces(
            cfg(
                f"rollouts/model_{model_name}_train_{train_dataset_name}_test_bash_codeforces_overwrite"
            ),
            reward_hacking_prompt="nothing",
            codeforces_dataset=load_taco_dataset(
                min_total_test_length=4096, difficulty="VERY_HARD"
            ),
            max_tests_per_split=256,
            reset_public_tests=False,
        )
        bash_codeforces_special_case_dataset_builder = bash_codeforces(
            cfg(
                f"rollouts/model_{model_name}_train_{train_dataset_name}_test_bash_codeforces_special_case"
            ),
            reward_hacking_prompt="nothing",
            codeforces_dataset=load_taco_dataset(difficulty="VERY_HARD"),
            max_tests_per_split=2,
            reset_public_tests=True,
        )
        style_dataset_builder = DatasetMixer(
            inner_builders=[
                style_environment(
                    cfg(f"rollouts/model_{model_name}_train_{train_dataset_name}_test_style"),
                    "nothing",
                    "none",
                    n_data_repetitions=999,
                ),
                style_environment(
                    cfg(f"rollouts/model_{model_name}_train_{train_dataset_name}_test_style"),
                    "nothing",
                    "contradictory",
                    n_data_repetitions=999,
                ),
                style_environment(
                    cfg(f"rollouts/model_{model_name}_train_{train_dataset_name}_test_style"),
                    "nothing",
                    "irrelevant",
                    n_data_repetitions=999,
                ),
                style_environment(
                    cfg(f"rollouts/model_{model_name}_train_{train_dataset_name}_test_style"),
                    "nothing",
                    "consistent",
                    n_data_repetitions=999,
                ),
            ],
        )
        ae_dataset_builder = ae(
            cfg(f"rollouts/model_{model_name}_train_{train_dataset_name}_test_ae"),
            reward_hacking_prompt="nothing",
            reset_public_tests=True,
        )

        for eval_dataset_name, dataset_builder in [
            ("bash_codeforces_overwrite", bash_codeforces_overwrite_dataset_builder),
            ("bash_codeforces_special_case", bash_codeforces_special_case_dataset_builder),
            ("style", style_dataset_builder),
            # ("ae", ae_dataset_builder),
        ]:
            sampling_client = service_client.create_sampling_client(model_path=sampler_path)
            _, dataset = await dataset_builder()
            env_builders = dataset.get_batch(0)

            rewards = await eval_environment(
                env_builders, sampling_client, max_tokens=cfg("").max_completion_tokens
            )
            print(f"{model_name=} {train_dataset_name=} {eval_dataset_name=} {mean(rewards)=}")


if __name__ == "__main__":
    build()
    # asyncio.run(main())
