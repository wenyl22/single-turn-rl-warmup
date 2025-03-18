# delete a huggingface repo
import huggingface_hub
def delete_repo(model_name):
    huggingface_hub.delete_repo(repo_id=model_name)

if __name__ == "__main__":
    model_name = "wenyl/Freeway-GRPO-Qwen-2.5-0.5B-Instruct"
    delete_repo(model_name)