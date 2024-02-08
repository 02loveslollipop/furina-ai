# DiscordGPTChatbot-API
> A simple rest API for LLM text-to-text generation using hugginngface transformers and flask, specifically design to operate with [DiscordGPTChatbot](https://github.com/02loveslollipop/DiscordGPTChatBot).

## Introduction
This project is design to allow to avoid the need to use the OpenAI API and their model (gpt-3.5-turbo, gpt-4) and instead give the user the ability to use their own model and host it on their own server. Thus allwoing the user to have full control over the model and the data it is trained on, allowing a more secure, private and customisable experience. This project is specifically designed to work with [DiscordGPTChatbot](https://github.com/02loveslollipop/DiscordGPTChatBot) but can be used with any other project that requieres an API for a huggeingface LLM model.

## Requirements

1. The project is designed to work in a Condda environment, but it should work in any other environment.

2. A huggeingface transformer model, the model should be a language model, and should be able to generate text. You can have it locally or in huggingface hub.

3. It's recommended to have a CUDA capable device to speed up the model inference.

4. As this project requires `bitsandbytes` this project should only work in a linux environment. But we have not tested it in other environments.

## Quick Setup

1. Clone the repository
```bash
git clone https://github.com/02loveslollipop/DiscordGPTChatBot.git
```

2. Create the conda environment with the `environment.yml` file
```bash
conda env create -f environment.yml
```

3. Activate the conda environment
```bash
conda activate DiscordGPTChatbot-API
```

4. Copy the `example_config.yml` and rename it to `config.yml`
```bash
cp example_config.yml config.yml
```

5. Edit the `config.yml` file to your needs, you will need to add the path to your model or the name of the model in huggeingface hub.

6. Run the server
```bash
python app.py
```

7. Test the server
```bash
curl --header "token: your-token" --header "request: what is a reference in object oriented programming??" http://your-host.domain:5000/generate
```

## Known Issues

1. The server is not secure, it's recommended to use a reverse proxy with a secure connection.

2. The server is not designed to handle multiple requests at the same time. A queue system is being implemented.

## License

This project is licensed under the MIT License
