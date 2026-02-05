from tinker_cookbook.rl.envs import (
    synthetic_env,
    swe_fixer_env,
    ae_env,
    bash_codeforces_env,
    bad_sandbox_env_with_tools,
)
from tinker_cookbook.rl.envs.aghyad_envs import omit_description_env

import asyncio


def build_everything(synthetic_dataset_path: str) -> None:
    print("building docker image for omit description env...")
    omit_description_env.build_docker_image()
    print("building docker image for resource bad sandbox env...")
    bad_sandbox_env_with_tools.build_docker_image()
    print("building docker image for bash codeforces env...")
    bash_codeforces_env.build_docker_image()
    print("building docker images for bash synthetic env..")
    synthetic_env.build_docker_images(
        synthetic_env.load_synthetic_env_dataset(synthetic_dataset_path)
    )
    print("building docker images for ae env...")
    asyncio.run(
        ae_env.build_docker_image(
            ae_env.load_ae_dataset_from_json("data/ae.json", max_datapoints=None)
        )
    )
    print("building docker images for swe fixer env...")
    swe_fixer_env.build_docker_images(swe_fixer_env.load_swe_fixer_dataset())
    print("done building all docker images")


def main() -> None:
    pass


if __name__ == "__main__":
    main()
