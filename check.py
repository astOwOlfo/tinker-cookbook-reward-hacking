import tinker
from dotenv import load_dotenv

load_dotenv()

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# checkpoint = "tinker://f3fbc2e3-b202-5328-a7c7-4f6763415e28:train:0/sampler_weights/0000100"
checkpoint = "tinker://f3fbc2e3-b202-5328-a7c7-4f6763415e28:train:0/sampler_weights/000100"

try:
    rest_client.get_weights_info_by_tinker_path(checkpoint).result()
    print("exists")
except Exception:
    print("doesn't exist")
