"""Chainlit app to demonstrate Langchain with OpenAI's LLMs.
This app is designed to be run with Chainlit, a framework for building
interactive applications using Langchain and OpenAI's LLMs.
The app initializes a chat session with an LLM, sets up a prompt template,
and processes user messages to generate responses using the LLM.
"""
# app.py

import os
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

"""Set up the models and prompt templates for the Chainlit app.
This function is called when the chat session starts.
It initializes the LLM, sets up the prompt template, and creates a chain
to process user messages.
The function is decorated with @cl.on_chat_start to indicate that it
should be called when a new chat session begins.
The function uses the ChatOpenAI model from Langchain to create a chat
session with OpenAI's LLM.
The prompt template is created using the ChatPromptTemplate class,
which allows for dynamic injection of user input into the prompt.
The LLMChain class is used to create a chain that processes user messages
and generates responses using the LLM.
The function also sets up a callback handler to stream the LLM's
responses back to the user in real-time.
The function is decorated with @cl.on_message to indicate that it
should be called when a new message is received in the chat session.
The function uses the LangchainCallbackHandler to handle the streaming
responses from the LLM and send them back to the user.
"""
@cl.on_chat_start
async def on_chat_start():
    ##########################################################################
    # Exercise 1a:
    # Our Chainlit app should initialize the LLM chat via Langchain at the
    # start of a chat session.
    #
    # First, we need to choose an LLM from OpenAI's list of models. Remember
    # to set streaming=True for streaming tokens
    ##########################################################################
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        streaming=True,
        openai_api_key=openai_api_key
    )

    ##########################################################################
    # Exercise 1b:
    # Next, we will need to set the prompt template for chat. Prompt templates
    # is how we set prompts and then inject informations into the prompt.
    #
    # Please create the prompt template using ChatPromptTemplate. Use variable
    # name "question" as the variable in the template.
    # Refer to the documentation listed in the README.md file for reference.
    ##########################################################################
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Chainlit GPT, a helpful assistant.",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )
    ##########################################################################
    # Exercise 1c:
    # Now we have model and prompt, let's build our Chain. A Chain is one or a
    # series of LLM calls.We will use the default StrOutputParser to parse the
    # LLM outputs.
    ##########################################################################
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

    # We are saving the chain in user_session, so we do not have to rebuild
    # it every single time.
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: LLMChain
    if chain is None:
        await cl.Message(content="Error: Chain not initialized.").send()
        return

    ##########################################################################
    # Exercise 1d:
    # Everytime we receive a new user message, we will get the chain from
    # user_session. We will run the chain with user's question and return LLM
    # response to the user.
    ##########################################################################
    response = await chain.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()
