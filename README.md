
# Art Critique Bot

## Analyze artwork using GPT Vision and LLM

Art Critique Bot is an app that uses GPT Vision (See: [OpenAI Platform](https://platform.openai.com/docs/guides/vision)) to identify artwork from images and AI language models like GPT-4 to provide detailed critiques of paintings, drawings, and other visual art forms. Users can upload an image of a piece for review, and the app will generate an analysis of the artwork covering composition, use of color, brushwork/texture, emotional impact, originality/creativity, and recommendations of similar artists and paintings.

Can we really use AI as an art critic or expert? Yes and No. This app is a fun experiment to see how far along GPT Vision has come, and the results are generally very good and surprisingly detailed. But, if you are looking to analyze and critique artwork in a more official capacity feel free to defer to art historians, experts, and enthusiasts.



## Tech

**Uses**:

* Python

* Streamlit (https://streamlit.io/)

* OpenAI - GPT-4 with Vision (GPT-4V or gpt-4-vision-preview - [OpenAI Platform](https://platform.openai.com/docs/guides/vision))

## Demo URL

* Streamlit - https://art-critique-bot-iubu3ncjq9mar7njrxrcx7.streamlit.app/ (requires OpenAI API Key)

## Development Environment

### Running the app

To start the app, run:

```
streamlit run streamlit_art.py
```

## Docker

A docker image is publicly available by pulling tejasvijain09/art-critique-bot:0.1 and on  [Dockerhub](https://hub.docker.com/repository/docker/tejasvijain09/art-critique-bot/general)

## Building the Docker Image

* Dockerfile based on instructions from https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker

* Pipreqs (https://pypi.org/project/pipreqs/) used to generate requirements.txt


```
# build an image with the 'latest' tag
make build

# build an image with a custom tag (e.g. 0.1)
make build TAG=<enter your custom tag>
```   

## Running the Docker Image

* Docker public repo at https://hub.docker.com/r/tejasvijain09/art-critique-bot


```
## Default Docker Run
docker run -p 8080:8501 tejasvijain09/art-critique-bot:0.1

## Run with custom eternal port
docker run -p 8080:8501 tejasvijain09/art-critique-bot:0.1

## Run with custom ENV variables containing the OPENAI_API_KEY var
docker run -p 8080:8501 --env-file .env tejasvijain09/art-critique-bot:0.1
```   

## Running on Heroku

This app has been tested on Heroku (https://heroku.com/) deploying as a Docker image and running with the Python buildpack.
For the application name, I used artcritiquebot-tj in this example, but you can use any name you like.
```
# Create a new Heroku app
heroku create  artcritiquebot-tj

# Login to the container registry
heroku container:login

# Push the container
heroku container:push web -a artcritiquebot-tj

# Release the container
heroku container:release web -a artcritiquebot-tj

# Note: Make sure to set PORT to 80 var under Settings > Config Vars
# And, if you want to use a default key make sure to set OPENAI_API_KEY 

# Open with the public URL
heroku open -a artcritiquebot-tj
```

## Known Issues

- On occasion, GPT-4V does not like to provide artist details, including similar artists or paintings (e.g. while analyzing Mona Lisa, the following is returned "*I cannot name the artist, but this painting is one of the most recognized works in the world.*").  Often times, simply reloading and trying again will return successful results.


## Version History

- 0.2 - 29/03/25 - Improved error handling for GPT-4V API calls and image results
- 0.1 - 18/03/25 - Initial release

## About

Analyze artwork using GPT Vision and LLM

This tool is a work in progress. You can contribute to the project on GitHub ([tejasvijain09/art-critique-bot · GitHub](https://github.com/tejasvijain09/art-critique-bot)) with your feedback and suggestions💡.

Created by Tejasvi Jain ([https://tejasvijain.com](https://tejasvijain.com/) & [https://tejasvijain.com](https://tejasvijain.com/)).
